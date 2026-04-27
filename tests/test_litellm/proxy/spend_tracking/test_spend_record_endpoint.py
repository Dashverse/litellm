"""Tests for POST /spend/record endpoint."""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import (
    SpendRecordRequest,
    SpendRecordResponse,
    UserAPIKeyAuth,
)
from litellm.proxy.spend_tracking.spend_record_endpoint import (
    _build_spend_log_payload,
    _calculate_cost_for_record,
    _is_duplicate_request_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(**overrides) -> SpendRecordRequest:
    defaults = {
        "request_id": "test-001",
        "model": "gpt-4",
        "call_type": "completion",
        "custom_llm_provider": "openai",
        "startTime": "2026-04-21T10:00:00+00:00",
        "endTime": "2026-04-21T10:00:05+00:00",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
    defaults.update(overrides)
    return SpendRecordRequest(**defaults)


def _make_auth(**overrides) -> UserAPIKeyAuth:
    defaults = {
        "api_key": "hashed-key-abc",
        "user_id": "user-1",
        "team_id": "team-1",
        "org_id": "org-1",
        "key_alias": "my-key",
        "team_alias": "my-team",
    }
    defaults.update(overrides)
    return UserAPIKeyAuth(**defaults)


# ---------------------------------------------------------------------------
# Cost calculation tests
# ---------------------------------------------------------------------------


def test_cost_calculation_text_completion():
    """Text model cost should be > 0 when tokens are provided and model is in cost map."""
    record = _make_record(
        model="gpt-4",
        custom_llm_provider="openai",
        prompt_tokens=100,
        completion_tokens=50,
    )
    cost = _calculate_cost_for_record(record)
    assert cost > 0, f"Expected non-zero cost for gpt-4, got {cost}"


def test_cost_calculation_builds_completion_response():
    """Verify completion_cost receives a completion_response dict with usage."""
    record = _make_record(prompt_tokens=100, completion_tokens=50)
    with patch(
        "litellm.proxy.spend_tracking.spend_record_endpoint.completion_cost",
        return_value=0.0045,
    ) as mock_cc:
        cost = _calculate_cost_for_record(record)
        assert cost == 0.0045
        mock_cc.assert_called_once()
        call_kwargs = mock_cc.call_args.kwargs
        # Verify completion_response was built with usage
        cr = call_kwargs["completion_response"]
        assert cr["usage"]["prompt_tokens"] == 100
        assert cr["usage"]["completion_tokens"] == 50


def test_cost_calculation_image_gen_builds_image_response():
    """Image generation should build an ImageResponse for completion_cost."""
    from litellm.types.utils import ImageResponse

    record = _make_record(
        model="fal_ai/flux-2/klein/9b",
        call_type="image_generation",
        custom_llm_provider="fal_ai",
        n=1,
        size="1024x1024",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )
    with patch(
        "litellm.proxy.spend_tracking.spend_record_endpoint.completion_cost",
        return_value=0.01,
    ) as mock_cc:
        cost = _calculate_cost_for_record(record)
        assert cost == 0.01
        call_kwargs = mock_cc.call_args.kwargs
        # Verify an ImageResponse was built
        cr = call_kwargs["completion_response"]
        assert isinstance(cr, ImageResponse)
        assert len(cr.data) == 1
        assert call_kwargs["size"] == "1024x1024"
        assert call_kwargs["call_type"] == "image_generation"


def test_cost_calculation_fal_image_real():
    """FAL image model cost should be > 0 using real completion_cost with local cost map."""
    import litellm

    # Ensure our custom model is in the cost map (it's in the local backup but
    # may not be in the remote upstream map fetched at import time)
    if "fal_ai/flux-2/klein/9b" not in litellm.model_cost:
        litellm.model_cost["fal_ai/flux-2/klein/9b"] = {
            "litellm_provider": "fal_ai",
            "mode": "image_generation",
            "pricing_basis": "PER_MEGAPIXEL",
            "output_cost_per_megapixel": 0.01,
            "output_cost_per_image": 0.01,
        }
    record = _make_record(
        model="fal_ai/flux-2/klein/9b",
        call_type="image_generation",
        custom_llm_provider="fal_ai",
        n=1,
        size="1024x1024",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )
    cost = _calculate_cost_for_record(record)
    assert cost > 0, f"Expected non-zero cost for fal_ai/flux-2/klein/9b, got {cost}"


def test_caller_provided_spend():
    """When caller provides spend, it bypasses calculation."""
    record = _make_record(spend=0.05)
    cost = _calculate_cost_for_record(record)
    assert cost == 0.05


def test_unknown_model_zero_cost():
    """Unrecognized model returns spend=0.0, no exception."""
    record = _make_record(
        model="unknown/nonexistent-model-xyz",
        custom_llm_provider="unknown",
    )
    cost = _calculate_cost_for_record(record)
    assert cost == 0.0


# ---------------------------------------------------------------------------
# Auth field tests
# ---------------------------------------------------------------------------


def test_auth_fields_from_api_key():
    """user/team/org come from auth, not request body."""
    record = _make_record(
        metadata={"user_id": "spoofed-user"},
        end_user="end-user-1",
    )
    auth = _make_auth(user_id="real-user", team_id="real-team", org_id="real-org")
    payload = _build_spend_log_payload(record, 0.01, auth)
    assert payload["user"] == "real-user"
    assert payload["team_id"] == "real-team"
    assert payload["organization_id"] == "real-org"
    assert payload["api_key"] == "hashed-key-abc"
    # end_user comes from request (not auth)
    assert payload["end_user"] == "end-user-1"


# ---------------------------------------------------------------------------
# Payload construction tests
# ---------------------------------------------------------------------------


def test_payload_construction():
    """SpendLogsPayload fields are correct."""
    record = _make_record(
        api_base="https://api.openai.com",
        request_tags=["tag1", "tag2"],
        cache_hit=True,
        status="success",
    )
    auth = _make_auth()
    payload = _build_spend_log_payload(record, 0.01, auth)

    assert payload["request_id"] == "test-001"
    assert payload["model"] == "gpt-4"
    assert payload["call_type"] == "completion"
    assert payload["custom_llm_provider"] == "openai"
    assert payload["spend"] == 0.01
    assert payload["prompt_tokens"] == 100
    assert payload["completion_tokens"] == 50
    assert payload["total_tokens"] == 150
    assert payload["api_base"] == "https://api.openai.com"
    assert payload["cache_hit"] == "True"
    assert payload["status"] == "success"
    assert json.loads(payload["request_tags"]) == ["tag1", "tag2"]
    assert isinstance(payload["startTime"], datetime)
    assert isinstance(payload["endTime"], datetime)
    assert payload["request_duration_ms"] == 5000

    # Metadata should contain auth info
    metadata = json.loads(payload["metadata"])
    assert metadata["user_api_key"] == "hashed-key-abc"
    assert metadata["user_api_key_team_id"] == "team-1"
    assert metadata["usage_object"]["prompt_tokens"] == 100
    assert metadata["cost_breakdown"]["total_cost"] == 0.01


def test_request_response_stored():
    """request and response bodies stored in payload messages, response, proxy_server_request fields."""
    req_body = {"prompt": "A cute cat", "image_size": "square_hd"}
    resp_body = {
        "data": [{"url": "https://example.com/img.png", "b64_json": None}],
        "created": 1776143113,
    }
    record = _make_record(request=req_body, response=resp_body)
    auth = _make_auth()
    payload = _build_spend_log_payload(record, 0.01, auth)

    # messages stores the request body
    assert json.loads(payload["messages"]) == req_body
    # response stores the response body
    assert json.loads(payload["response"]) == resp_body
    # proxy_server_request also stores the request body
    assert json.loads(payload["proxy_server_request"]) == req_body


def test_payload_none_request_response():
    """When request/response are None, payload fields default to '{}' for Prisma compatibility."""
    record = _make_record(request=None, response=None)
    auth = _make_auth()
    payload = _build_spend_log_payload(record, 0.01, auth)

    assert payload["messages"] == "{}"
    assert payload["response"] == "{}"
    assert payload["proxy_server_request"] == "{}"


# ---------------------------------------------------------------------------
# Insert and aggregate tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_and_aggregate_called():
    """Both _insert_spend_log_to_db and _batch_database_updates called."""
    record = _make_record()

    mock_prisma = MagicMock()
    mock_prisma.spend_log_transactions = []
    mock_prisma._spend_log_transactions_lock = asyncio.Lock()

    mock_db_writer = MagicMock()
    mock_db_writer._insert_spend_log_to_db = AsyncMock(return_value=mock_prisma)
    mock_db_writer._batch_database_updates = AsyncMock()

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.db_spend_update_writer = mock_db_writer

    auth = _make_auth()

    with patch(
        "litellm.proxy.spend_tracking.spend_record_endpoint.completion_cost",
        return_value=0.01,
    ), patch.dict(
        "litellm.proxy.proxy_server.__dict__",
        {
            "prisma_client": mock_prisma,
            "proxy_logging_obj": mock_proxy_logging,
            "user_api_key_cache": MagicMock(),
            "litellm_proxy_budget_name": "litellm-proxy-budget",
        },
    ):
        from litellm.proxy.spend_tracking.spend_record_endpoint import (
            spend_record,
        )

        mock_request = MagicMock()
        result = await spend_record(record, mock_request, auth)

        assert result.recorded is True
        assert result.spend == 0.01
        mock_db_writer._insert_spend_log_to_db.assert_called_once()

        # Let the background task run
        await asyncio.sleep(0.05)
        mock_db_writer._batch_database_updates.assert_called_once()


# ---------------------------------------------------------------------------
# Duplicate detection tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_request_id_detected():
    """_is_duplicate_request_id returns True when request_id exists in pending transactions."""
    mock_prisma = MagicMock()
    mock_prisma.spend_log_transactions = [{"request_id": "dup-001", "spend": 0.01}]
    mock_prisma._spend_log_transactions_lock = asyncio.Lock()

    assert await _is_duplicate_request_id("dup-001", mock_prisma) is True
    assert await _is_duplicate_request_id("new-001", mock_prisma) is False


@pytest.mark.asyncio
async def test_duplicate_request_id_skipped():
    """Same request_id returns recorded=False and no DB calls."""
    record = _make_record(request_id="dup-001")

    mock_prisma = MagicMock()
    mock_prisma.spend_log_transactions = [{"request_id": "dup-001", "spend": 0.01}]
    mock_prisma._spend_log_transactions_lock = asyncio.Lock()

    mock_db_writer = MagicMock()
    mock_db_writer._insert_spend_log_to_db = AsyncMock()
    mock_db_writer._batch_database_updates = AsyncMock()

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.db_spend_update_writer = mock_db_writer

    auth = _make_auth()

    with patch.dict(
        "litellm.proxy.proxy_server.__dict__",
        {
            "prisma_client": mock_prisma,
            "proxy_logging_obj": mock_proxy_logging,
            "user_api_key_cache": MagicMock(),
            "litellm_proxy_budget_name": "litellm-proxy-budget",
        },
    ):
        from litellm.proxy.spend_tracking.spend_record_endpoint import (
            spend_record,
        )

        mock_request = MagicMock()
        result = await spend_record(record, mock_request, auth)

        assert result.recorded is False
        assert result.spend == 0.0
        mock_db_writer._insert_spend_log_to_db.assert_not_called()
