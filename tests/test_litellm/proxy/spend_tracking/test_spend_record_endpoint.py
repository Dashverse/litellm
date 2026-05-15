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
    _enrich_record_from_payloads,
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
# Payload enrichment tests (_enrich_record_from_payloads)
# ---------------------------------------------------------------------------


def test_enrich_tokens_from_response_usage():
    """Tokens extracted from response.usage when not provided explicitly."""
    record = _make_record(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        response={
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 200,
                "total_tokens": 700,
            },
            "choices": [{"message": {"content": "hello"}}],
        },
    )
    _enrich_record_from_payloads(record)
    assert record.prompt_tokens == 500
    assert record.completion_tokens == 200
    assert record.total_tokens == 700


def test_enrich_tokens_not_overwritten():
    """Explicit tokens are not overwritten by response.usage."""
    record = _make_record(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        response={"usage": {"prompt_tokens": 999, "completion_tokens": 888}},
    )
    _enrich_record_from_payloads(record)
    assert record.prompt_tokens == 100
    assert record.completion_tokens == 50


def test_enrich_n_from_response_data():
    """Image count extracted from response.data list."""
    record = _make_record(
        call_type="image_generation",
        n=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        response={"data": [{"url": "a.png"}, {"url": "b.png"}]},
    )
    _enrich_record_from_payloads(record)
    assert record.n == 2


def test_enrich_n_from_response_images():
    """Image count extracted from FAL-style response.images list."""
    record = _make_record(
        call_type="image_generation",
        n=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        response={"images": [{"url": "a.png", "width": 1024, "height": 1024}]},
    )
    _enrich_record_from_payloads(record)
    assert record.n == 1


def test_enrich_n_defaults_to_1_for_images():
    """Image gen with no data/images in response defaults n=1."""
    record = _make_record(
        call_type="image_generation",
        n=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        response={"status": "success"},
    )
    _enrich_record_from_payloads(record)
    assert record.n == 1


def test_enrich_n_not_set_for_completion():
    """Non-image call types don't get n set."""
    record = _make_record(
        call_type="completion",
        n=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )
    _enrich_record_from_payloads(record)
    assert record.n is None


def test_enrich_size_from_request():
    """Size extracted from request.size (OpenAI format)."""
    record = _make_record(
        size=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        request={"prompt": "cat", "size": "1024x1024"},
    )
    _enrich_record_from_payloads(record)
    assert record.size == "1024x1024"


def test_enrich_size_from_fal_image_size():
    """Size extracted from FAL request.image_size and mapped to pixels."""
    record = _make_record(
        size=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        request={"prompt": "cat", "image_size": "square_hd"},
    )
    _enrich_record_from_payloads(record)
    assert record.size == "1024x1024"


def test_enrich_size_from_width_height():
    """Size extracted from request.width/height (SimpliSmart/BytePlus)."""
    record = _make_record(
        size=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        request={"prompt": "cat", "width": 768, "height": 1024},
    )
    _enrich_record_from_payloads(record)
    assert record.size == "768x1024"


def test_enrich_quality_from_request():
    """Quality extracted from request.quality."""
    record = _make_record(
        quality=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        request={"prompt": "cat", "quality": "hd"},
    )
    _enrich_record_from_payloads(record)
    assert record.quality == "hd"


def test_enrich_no_payloads():
    """Enrichment is a no-op when request/response are None."""
    record = _make_record(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        request=None,
        response=None,
    )
    _enrich_record_from_payloads(record)
    assert record.prompt_tokens == 0
    assert record.n is None
    assert record.size is None


def test_cost_calc_uses_enriched_tokens():
    """Cost calculation uses tokens enriched from response payload."""
    record = _make_record(
        model="gpt-4",
        custom_llm_provider="openai",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        response={
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        },
    )
    cost = _calculate_cost_for_record(record)
    assert cost > 0, f"Expected non-zero cost after enrichment, got {cost}"


def test_cost_calc_uses_enriched_n_for_images():
    """Image cost uses n enriched from response.data."""
    record = _make_record(
        call_type="image_generation",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        n=None,
        size="1024x1024",
        response={"data": [{"url": "a.png"}, {"url": "b.png"}]},
    )
    with patch(
        "litellm.proxy.spend_tracking.spend_record_endpoint.completion_cost",
        return_value=0.02,
    ) as mock_cc:
        cost = _calculate_cost_for_record(record)
        assert cost == 0.02
        # n should have been enriched to 2
        assert record.n == 2
        cr = mock_cc.call_args.kwargs["completion_response"]
        from litellm.types.utils import ImageResponse

        assert isinstance(cr, ImageResponse)
        assert len(cr.data) == 2


# ---------------------------------------------------------------------------
# Audio cost calculation tests (TTS + STT)
# ---------------------------------------------------------------------------


def test_cost_calc_tts_character_based_uses_padded_prompt():
    """TTS character-priced models (tts-1) — char count from response['characters']
    is converted to a non-whitespace prompt of that length, since completion_cost
    derives prompt_characters via _count_characters(text=prompt)."""
    record = _make_record(
        model="tts-1",
        call_type="aspeech",
        custom_llm_provider="openai",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        response={"characters": 42},
    )
    with patch(
        "litellm.proxy.spend_tracking.spend_record_endpoint.completion_cost",
        return_value=0.00063,
    ) as mock_cc:
        cost = _calculate_cost_for_record(record)
        assert cost == 0.00063
        call_kwargs = mock_cc.call_args.kwargs
        assert call_kwargs["call_type"] == "aspeech"
        prompt = call_kwargs["prompt"]
        assert len(prompt) == 42
        assert prompt.strip() == prompt  # no whitespace — survives _count_characters


def test_cost_calc_tts_token_based_passes_usage():
    """TTS token-priced models (gpt-4o-mini-tts) — usage dict flows through."""
    record = _make_record(
        model="gpt-4o-mini-tts",
        call_type="aspeech",
        custom_llm_provider="openai",
        prompt_tokens=10,
        completion_tokens=50,
        total_tokens=60,
        response={"characters": 0},
    )
    with patch(
        "litellm.proxy.spend_tracking.spend_record_endpoint.completion_cost",
        return_value=0.0007,
    ) as mock_cc:
        cost = _calculate_cost_for_record(record)
        assert cost == 0.0007
        cr = mock_cc.call_args.kwargs["completion_response"]
        assert cr["usage"]["prompt_tokens"] == 10
        assert cr["usage"]["completion_tokens"] == 50


def test_cost_calc_tts_missing_characters_uses_zero_length_prompt():
    """TTS with no `characters` in response should not raise; passes empty prompt."""
    record = _make_record(
        model="tts-1",
        call_type="aspeech",
        custom_llm_provider="openai",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        response={},
    )
    with patch(
        "litellm.proxy.spend_tracking.spend_record_endpoint.completion_cost",
        return_value=0.0,
    ) as mock_cc:
        cost = _calculate_cost_for_record(record)
        assert cost == 0.0
        assert mock_cc.call_args.kwargs["prompt"] == ""


def test_cost_calc_stt_duration_based_builds_transcription_response():
    """STT duration-priced models (whisper-1) — TranscriptionResponse with .duration."""
    from litellm.types.utils import TranscriptionResponse

    record = _make_record(
        model="whisper-1",
        call_type="atranscription",
        custom_llm_provider="openai",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        duration_seconds=10.0,
    )
    with patch(
        "litellm.proxy.spend_tracking.spend_record_endpoint.completion_cost",
        return_value=0.001,
    ) as mock_cc:
        cost = _calculate_cost_for_record(record)
        assert cost == 0.001
        cr = mock_cc.call_args.kwargs["completion_response"]
        assert isinstance(cr, TranscriptionResponse)
        assert cr.duration == 10.0
        assert mock_cc.call_args.kwargs["call_type"] == "atranscription"


def test_cost_calc_stt_token_based_passes_usage():
    """STT token-priced models (gpt-4o-transcribe) — TranscriptionResponse with Usage."""
    from litellm.types.utils import TranscriptionResponse, Usage

    record = _make_record(
        model="gpt-4o-transcribe",
        call_type="atranscription",
        custom_llm_provider="openai",
        prompt_tokens=14,
        completion_tokens=45,
        total_tokens=59,
    )
    with patch(
        "litellm.proxy.spend_tracking.spend_record_endpoint.completion_cost",
        return_value=0.00054,
    ) as mock_cc:
        cost = _calculate_cost_for_record(record)
        assert cost == 0.00054
        cr = mock_cc.call_args.kwargs["completion_response"]
        assert isinstance(cr, TranscriptionResponse)
        assert isinstance(cr.usage, Usage)
        assert cr.usage.prompt_tokens == 14
        assert cr.usage.completion_tokens == 45


def test_enrich_stt_duration_from_response():
    """STT duration_seconds extracted from response.duration when not set."""
    record = _make_record(
        call_type="atranscription",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        response={"duration": 42.7, "text": "hello there"},
    )
    _enrich_record_from_payloads(record)
    assert record.duration_seconds == 42.7


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
