"""POST /spend/record — record externally-completed inference for cost tracking."""

import asyncio
import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from litellm import completion_cost
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    SpendLogsMetadata,
    SpendLogsPayload,
    SpendRecordRequest,
    SpendRecordResponse,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.proxy.proxy_server import PrismaClient
else:
    PrismaClient = Any

router = APIRouter()

_IMAGE_CALL_TYPES = {
    "image_generation",
    "aimage_generation",
    "image_edit",
    "aimage_edit",
}


def _calculate_cost_for_record(record: SpendRecordRequest) -> float:
    """Return the spend for a record, using caller-provided value or computing via completion_cost()."""
    if record.spend is not None:
        return record.spend

    try:
        completion_response: Any = None

        if record.call_type in _IMAGE_CALL_TYPES:
            # Build a synthetic ImageResponse so completion_cost's isinstance check passes
            n = record.n or 1
            completion_response = ImageResponse(
                created=0,
                data=[ImageObject(url="placeholder") for _ in range(n)],
            )
            if record.size:
                completion_response.size = record.size
            if record.quality:
                completion_response.quality = record.quality
        else:
            # Build a dict with usage info for token-based calls
            completion_response = {
                "usage": {
                    "prompt_tokens": record.prompt_tokens,
                    "completion_tokens": record.completion_tokens,
                    "total_tokens": record.total_tokens,
                },
                "model": record.model,
            }

        cost = completion_cost(
            completion_response=completion_response,
            model=record.model,
            custom_llm_provider=record.custom_llm_provider,
            call_type=record.call_type,
            size=record.size,
            quality=record.quality,
            n=record.n,
        )
        return cost
    except Exception as e:
        verbose_proxy_logger.debug(
            "spend/record: cost calculation failed for model=%s: %s",
            record.model,
            str(e),
        )
        return 0.0


def _build_spend_log_payload(
    record: SpendRecordRequest,
    spend: float,
    user_api_key_dict: UserAPIKeyAuth,
) -> SpendLogsPayload:
    """Build a SpendLogsPayload from the record and auth context."""
    start_time = datetime.fromisoformat(record.startTime)
    end_time = datetime.fromisoformat(record.endTime)
    duration_ms = int((end_time - start_time).total_seconds() * 1000)

    hashed_token = getattr(user_api_key_dict, "api_key", None) or ""
    user_id = getattr(user_api_key_dict, "user_id", None) or ""
    team_id = getattr(user_api_key_dict, "team_id", None)
    org_id = getattr(user_api_key_dict, "org_id", None)
    key_alias = getattr(user_api_key_dict, "key_alias", None)
    team_alias = getattr(user_api_key_dict, "team_alias", None)

    # Serialize request and response for storage
    # Default to "{}" (not None) — Prisma's create_many rejects None for Json? fields
    messages_value: Any = "{}"
    if record.request is not None:
        messages_value = (
            json.dumps(record.request)
            if not isinstance(record.request, str)
            else record.request
        )

    response_value: Any = "{}"
    if record.response is not None:
        response_value = (
            json.dumps(record.response)
            if not isinstance(record.response, str)
            else record.response
        )

    proxy_server_request_value: Optional[str] = "{}"
    if record.request is not None:
        proxy_server_request_value = (
            json.dumps(record.request)
            if not isinstance(record.request, str)
            else record.request
        )

    # Build metadata
    usage_object = {
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "total_tokens": record.total_tokens,
    }
    cost_breakdown = {"total_cost": spend}

    metadata: SpendLogsMetadata = {
        "additional_usage_values": None,
        "user_api_key": hashed_token,
        "user_api_key_alias": key_alias,
        "user_api_key_team_id": team_id,
        "user_api_key_project_id": None,
        "user_api_key_project_alias": None,
        "user_api_key_org_id": org_id,
        "user_api_key_user_id": user_id,
        "user_api_key_team_alias": team_alias,
        "spend_logs_metadata": record.metadata,
        "requester_ip_address": None,
        "applied_guardrails": None,
        "mcp_tool_call_metadata": None,
        "vector_store_request_metadata": None,
        "guardrail_information": None,
        "status": "success" if record.status == "success" else "failure",
        "proxy_server_request": proxy_server_request_value,
        "batch_models": None,
        "error_information": None,
        "usage_object": usage_object,
        "model_map_information": None,
        "cold_storage_object_key": None,
        "litellm_overhead_time_ms": None,
        "attempted_retries": None,
        "max_retries": None,
        "cost_breakdown": cost_breakdown,
    }

    payload: SpendLogsPayload = {
        "request_id": record.request_id,
        "call_type": record.call_type,
        "api_key": hashed_token,
        "spend": spend,
        "total_tokens": record.total_tokens,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "startTime": start_time,
        "endTime": end_time,
        "completionStartTime": None,
        "model": record.model,
        "model_id": None,
        "model_group": None,
        "mcp_namespaced_tool_name": None,
        "agent_id": None,
        "api_base": record.api_base or "",
        "user": user_id,
        "metadata": json.dumps(metadata),
        "cache_hit": "True" if record.cache_hit else "",
        "cache_key": "",
        "request_tags": json.dumps(record.request_tags or []),
        "team_id": team_id,
        "organization_id": org_id,
        "end_user": record.end_user,
        "requester_ip_address": None,
        "custom_llm_provider": record.custom_llm_provider,
        "messages": messages_value,
        "response": response_value,
        "proxy_server_request": proxy_server_request_value,
        "session_id": str(uuid.uuid4()),
        "request_duration_ms": duration_ms,
        "status": "success" if record.status == "success" else "failure",
    }
    return payload


async def _is_duplicate_request_id(request_id: str, prisma_client: Any) -> bool:
    """Check if request_id already exists in the pending spend log transactions."""
    async with prisma_client._spend_log_transactions_lock:
        for txn in prisma_client.spend_log_transactions:
            if isinstance(txn, dict) and txn.get("request_id") == request_id:
                return True
    return False


@router.post(
    "/spend/record",
    tags=["Budget & Spend Tracking"],
    response_model=SpendRecordResponse,
)
async def spend_record(
    record: SpendRecordRequest,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Record an externally-completed inference result for cost tracking.

    The caller sends the request/response from a direct provider call,
    and LiteLLM calculates cost and stores the spend log — identical to
    what the proxy would store if the call had gone through it.
    """
    from litellm.proxy.proxy_server import (
        litellm_proxy_budget_name,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail="Database not connected. Connect a database to your proxy.",
        )

    # Check for duplicate in pending transactions
    is_dup = await _is_duplicate_request_id(record.request_id, prisma_client)
    if is_dup:
        return SpendRecordResponse(
            request_id=record.request_id,
            spend=0.0,
            model=record.model,
            recorded=False,
        )

    # Calculate cost
    spend = _calculate_cost_for_record(record)

    # Build payload
    payload = _build_spend_log_payload(record, spend, user_api_key_dict)

    # Insert spend log
    db_writer = proxy_logging_obj.db_spend_update_writer
    await db_writer._insert_spend_log_to_db(payload, prisma_client)

    # Update key/user/team/org spend aggregates in background
    hashed_token = getattr(user_api_key_dict, "api_key", None)
    user_id = getattr(user_api_key_dict, "user_id", None)
    team_id = getattr(user_api_key_dict, "team_id", None)
    org_id = getattr(user_api_key_dict, "org_id", None)

    asyncio.create_task(
        db_writer._batch_database_updates(
            response_cost=spend,
            user_id=user_id,
            hashed_token=hashed_token,
            team_id=team_id,
            org_id=org_id,
            end_user_id=record.end_user,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            litellm_proxy_budget_name=litellm_proxy_budget_name,
            payload_copy=payload,
            request_tags=record.request_tags,
        )
    )

    return SpendRecordResponse(
        request_id=record.request_id,
        spend=spend,
        model=record.model,
        recorded=True,
    )
