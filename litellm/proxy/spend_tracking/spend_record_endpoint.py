"""POST /spend/record — record externally-completed inference for cost tracking."""

import asyncio
import json
import uuid
from datetime import datetime
from types import SimpleNamespace
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

_VIDEO_CALL_TYPES = {
    "video_generation",
    "create_video",
    "acreate_video",
    "video_remix",
    "avideo_remix",
}

# Map BytePlus ARK endpoint IDs to canonical model names
_BYTEPLUS_ARK_MODEL_MAP = {
    "dreamina-seedance-2-0-260128": "seedance-2.0",
    "dreamina-seedance-2-0-fast-260128": "seedance-2.0-fast",
    "ep-20260130153806-lrrh6": "seedance-v1.5-pro",
}

# BytePlus video: token-based pricing (USD per token)
_BYTEPLUS_COST_PER_TOKEN: dict[str, float] = {
    "seedance-2.0": 0.000007,
    "seedance-2.0-fast": 0.0000056,
    "seedance-v1.5-pro": 0.0000025,
}

# Local per-second pricing for video models not in the upstream cost map.
# Key: (provider, canonical_model, mode, has_audio) → USD per second.
_VIDEO_COST_PER_SECOND: dict[tuple[str, str, str, bool], float] = {
    # --- Kling direct API ---
    # Kling 3.0
    ("kling", "kling-v3", "pro", False): 0.08,
    ("kling", "kling-v3", "pro", True): 0.12,
    ("kling", "kling-v3", "std", False): 0.06,
    ("kling", "kling-v3", "std", True): 0.10,
    # Kling o3
    ("kling", "kling-v3-omni", "pro", False): 0.08,
    ("kling", "kling-v3-omni", "pro", True): 0.10,
    ("kling", "kling-v3-omni", "std", False): 0.06,
    ("kling", "kling-v3-omni", "std", True): 0.08,
    # Kling 2.6
    ("kling", "kling-v2-6", "pro", False): 0.08,
    ("kling", "kling-v2-6", "pro", True): 0.12,
    # Kling 2.1
    ("kling", "kling-v2-1", "std", False): 0.05,
    ("kling", "kling-v2-1", "std", True): 0.06,
    ("kling", "kling-v2-1", "pro", False): 0.098,
    ("kling", "kling-v2-1", "master", False): 0.20,
    ("kling", "kling-v2-1", "master", True): 0.24,
    # --- VEO (Vertex AI) ---
    ("vertex_ai", "veo-3.1-fast-generate-001", "default", False): 0.10,
    ("vertex_ai", "veo-3.1-fast-generate-001", "default", True): 0.15,
    ("vertex_ai", "veo-3.1-generate-001", "default", False): 0.10,
    ("vertex_ai", "veo-3.1-generate-001", "default", True): 0.15,
    ("vertex_ai", "veo-3.0-fast-generate-001", "default", False): 0.10,
    ("vertex_ai", "veo-3.0-fast-generate-001", "default", True): 0.15,
    ("vertex_ai", "veo-3.0-generate-001", "default", False): 0.10,
    ("vertex_ai", "veo-3.0-generate-001", "default", True): 0.15,
    ("vertex_ai", "veo-2.0-generate-001", "default", False): 0.10,
    ("vertex_ai", "veo-2.0-generate-001", "default", True): 0.15,
}

# Map FAL named image sizes to pixel dimensions
_FAL_SIZE_MAP = {
    "square_hd": "1024x1024",
    "square": "512x512",
    "landscape_4_3": "1280x960",
    "landscape_16_9": "1792x1024",
    "portrait_4_3": "960x1280",
    "portrait_16_9": "1024x1792",
}


def _enrich_record_from_payloads(record: SpendRecordRequest) -> None:
    """Extract tokens, n, size, quality from request/response bodies when not set explicitly.

    Mutates the record in-place so _calculate_cost_for_record can use the enriched fields.
    Handles response formats from OpenAI/Gemini, FAL, SimpliSmart, BytePlus, etc.
    """
    req = record.request if isinstance(record.request, dict) else {}
    resp = record.response if isinstance(record.response, dict) else {}

    # --- Tokens from response.usage ---
    if record.prompt_tokens == 0 and record.completion_tokens == 0:
        usage = resp.get("usage") or {}
        if isinstance(usage, dict) and usage:
            record.prompt_tokens = usage.get("prompt_tokens", 0) or 0
            record.completion_tokens = usage.get("completion_tokens", 0) or 0
            record.total_tokens = usage.get("total_tokens", 0) or (
                record.prompt_tokens + record.completion_tokens
            )

    # --- Image count (n) from response ---
    if record.n is None and record.call_type in _IMAGE_CALL_TYPES:
        data = resp.get("data")
        if isinstance(data, list) and data:
            record.n = len(data)
        else:
            # FAL direct responses use "images" key
            images = resp.get("images")
            if isinstance(images, list) and images:
                record.n = len(images)
        # Default to 1 for image gen if still unset
        if record.n is None:
            record.n = 1

    # --- Size from request body ---
    if record.size is None:
        # OpenAI/standard format
        size_val = req.get("size")
        if size_val:
            record.size = str(size_val)
        else:
            # FAL named size format (e.g., "square_hd", "landscape_16_9")
            image_size = req.get("image_size")
            if image_size:
                record.size = _FAL_SIZE_MAP.get(str(image_size), str(image_size))
            else:
                # Width/height format (SimpliSmart, BytePlus, etc.)
                w = req.get("width")
                h = req.get("height")
                if w and h:
                    record.size = f"{w}x{h}"

    # --- Quality from request ---
    if record.quality is None:
        quality_val = req.get("quality")
        if quality_val:
            record.quality = str(quality_val)

    # --- Video duration from request body ---
    if record.duration_seconds is None and record.call_type in _VIDEO_CALL_TYPES:
        # Try common field names used by video providers:
        #   "duration" (BytePlus, Kling, FAL), "seconds" (Azure Sora),
        #   "durationSeconds" in nested "params" (VEO/Vertex AI)
        dur_val = req.get("duration") or req.get("seconds")
        if dur_val is None:
            params = req.get("params") or req.get("parameters") or {}
            if isinstance(params, dict):
                dur_val = params.get("durationSeconds") or params.get(
                    "duration_seconds"
                )
        if dur_val is not None:
            try:
                record.duration_seconds = float(dur_val)
            except (ValueError, TypeError):
                pass


def _extract_video_cost_key(
    record: SpendRecordRequest,
) -> tuple[str, str, str, bool]:
    """Return (provider, canonical_model, mode, has_audio) for local pricing lookup."""
    req = record.request if isinstance(record.request, dict) else {}
    provider = record.custom_llm_provider or ""

    # Resolve canonical model name (strip provider prefix)
    if provider == "byteplus":
        raw = record.model.removeprefix("byteplus/")
        canonical = _BYTEPLUS_ARK_MODEL_MAP.get(raw, raw)
    elif provider == "kling":
        canonical = record.model.removeprefix("kling/")
    elif provider == "vertex_ai":
        canonical = record.model.removeprefix("vertex_ai/")
    else:
        canonical = (
            record.model.split("/", 1)[-1] if "/" in record.model else record.model
        )

    # Extract mode (Kling uses "mode" field; default for other providers)
    mode = req.get("mode", "default")

    # Extract audio flag
    has_audio = False
    if req.get("generate_audio") is True:  # BytePlus
        has_audio = True
    elif req.get("sound") in ("on", True, "true"):  # Kling
        has_audio = True

    return (provider, canonical, mode, has_audio)


def _calculate_cost_for_record(record: SpendRecordRequest) -> float:
    """Return the spend for a record, using caller-provided value or computing via completion_cost()."""
    if record.spend is not None:
        return record.spend

    # Failed requests should not incur cost
    if record.status == "failure":
        return 0.0

    # Enrich record fields from raw request/response payloads
    _enrich_record_from_payloads(record)

    try:
        if record.call_type in _VIDEO_CALL_TYPES:
            return _calculate_video_cost(record)

        completion_response: Any = None
        effective_call_type = record.call_type

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
            call_type=effective_call_type,
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


def _calculate_video_cost(record: SpendRecordRequest) -> float:
    """Calculate cost for video generation calls.

    Provider-specific strategies:
    - BytePlus: token-based pricing via completion_cost() with mapped model name
    - Kling/VEO: per-second pricing from local _VIDEO_COST_PER_SECOND table
    - Others (Sora, etc.): fallback to completion_cost() with duration_seconds
    """
    # --- BytePlus: token-based pricing ---
    if record.custom_llm_provider == "byteplus":
        raw = record.model.removeprefix("byteplus/")
        canonical = _BYTEPLUS_ARK_MODEL_MAP.get(raw, raw)
        cost_per_token = _BYTEPLUS_COST_PER_TOKEN.get(canonical)
        if cost_per_token is not None:
            prompt_cost = record.prompt_tokens * cost_per_token
            completion_cost_val = record.completion_tokens * cost_per_token
            return prompt_cost + completion_cost_val

    # --- Kling / VEO: per-second from local pricing table ---
    key = _extract_video_cost_key(record)
    rate = _VIDEO_COST_PER_SECOND.get(key)
    if rate is not None:
        duration = record.duration_seconds or 0.0
        return rate * duration

    # --- Fallback: completion_cost() for models in upstream cost map (Sora, etc.) ---
    duration = record.duration_seconds or 0.0
    completion_response = SimpleNamespace(
        usage={"duration_seconds": duration},
        model=record.model,
    )
    return completion_cost(
        completion_response=completion_response,
        model=record.model,
        custom_llm_provider=record.custom_llm_provider,
        call_type="create_video",
    )


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
