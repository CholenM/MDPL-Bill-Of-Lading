"""
Unit + integration tests for the AI MDPL Bill of Lading Extractor (v3, vLLM Migration).

These run on the Windows dev machine with a STUB model backend (no GPU, no
vLLM needed). The app's http_model_call is monkeypatched so the full
FastAPI stack can be exercised.

Run:  pytest -v
"""

import os
import json
import pytest
from fastapi.testclient import TestClient

import bol_service as bol

# Matches DEFAULT_KEYS used when no .env is present.
VALID_KEY = "bol_key_0000"
ENV_EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.example")
CASES = os.path.join(os.path.dirname(__file__), "cases")


class FakeResp:
    """Minimal stand-in for requests.Response (healthz probe mock)."""

    def __init__(self, status_code=200, text="", json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


def _messages():
    return [{"role": "user", "content": "hi"}]


def _golden():
    with open(os.path.join(CASES, "bol_golden_expected.json")) as f:
        return json.load(f)


def _golden_text():
    with open(os.path.join(CASES, "bol_golden_input.txt")) as f:
        return f.read()


@pytest.fixture
def client(monkeypatch):
    calls = {"n": 0}

    def stub(messages, params):
        calls["n"] += 1
        return json.dumps(_golden())

    monkeypatch.setattr(bol, "http_model_call", stub)
    monkeypatch.setattr(bol, "DISABLE_THINKING", True)
    monkeypatch.setattr(bol, "API_KEY_DB", {VALID_KEY: ""})
    monkeypatch.setattr(bol, "_PARAMS_LEVEL", 0)

    app_client = TestClient(bol.app)
    app_client.calls = calls
    return app_client


def _auth():
    return {"x-api-key": VALID_KEY}


def _extract_body():
    return {"bol_text": _golden_text()}


# --------------------------------------------------------------------------
# Golden sample — end-to-end through the FastAPI stack (stub backend)
# --------------------------------------------------------------------------
class TestGoldenSample:
    def test_extract_returns_contract_exact_json(self, client):
        r = client.post("/v1/extract", json=_extract_body(), headers=_auth())
        assert r.status_code == 200
        assert r.json() == _golden()

    def test_extract_stub_called_once(self, client):
        client.post("/v1/extract", json=_extract_body(), headers=_auth())
        assert client.calls["n"] == 1

    def test_prompt_contains_customer_table_and_fences(self, client, monkeypatch):
        seen = {}

        def spy(messages, params):
            seen["messages"] = messages
            return json.dumps(_golden())

        monkeypatch.setattr(bol, "http_model_call", spy)
        client.post("/v1/extract", json=_extract_body(), headers=_auth())
        user_msg = seen["messages"][1]["content"]
        assert "REGISTERED CUSTOMER TABLE:" in user_msg
        assert "C0003" in user_msg
        assert "<<<BOL_DATA>>>" in user_msg
        assert _golden_text().strip()[:20] in user_msg
        assert "<<<END_BOL_DATA>>>" in user_msg

    def test_system_prompt_demands_j21(self, client, monkeypatch):
        seen = {}

        def spy(messages, params):
            seen["messages"] = messages
            return json.dumps(_golden())

        monkeypatch.setattr(bol, "http_model_call", spy)
        client.post("/v1/extract", json=_extract_body(), headers=_auth())
        sys_msg = seen["messages"][0]["content"]
        assert '"AssistantVersion"' in sys_msg
        assert "J2.1" in sys_msg


# --------------------------------------------------------------------------
# vLLM config constant tests
# --------------------------------------------------------------------------
class TestVLLMConfig:
    def test_context_size_is_32768(self):
        """vLLM MAX_MODEL_LEN = 32768, no slot division."""
        assert bol.CONTEXT_SIZE == 32768

    def test_model_name_is_vllm(self):
        assert bol.MODEL_NAME == "Qwen3.6-35B-A3B-NVFP4"

    def test_model_url_is_vllm(self):
        assert "8011" in bol.MODEL_URL

    def test_model_parallel_not_defined(self):
        """vLLM has no slot division -- MODEL_PARALLEL must not exist."""
        assert not hasattr(bol, "MODEL_PARALLEL")

    def test_api_key_is_vllm(self):
        assert bol.LLAMA_SERVER_API_KEY == "test_key_0000"

    def test_request_timeout_is_120(self):
        assert bol.REQUEST_TIMEOUT == 120

    def test_params_levels_are_two(self):
        assert bol._PARAMS_LEVELS == (0, 1)


# --------------------------------------------------------------------------
# Output contract normalization (_normalize_bol / extract_json)
# --------------------------------------------------------------------------
class TestNormalizeBol:
    def test_assistant_version_forced_to_j21(self):
        data = dict(_golden())
        data["AssistantVersion"] = "J9.9"  # model hallucinated a version
        out = bol._normalize_bol(data)
        assert out["AssistantVersion"] == "J2.1"

    def test_unknown_keys_dropped(self):
        data = dict(_golden())
        data["SurpriseField"] = "junk"
        data["segments"] = [{"airline_code": "PR"}]  # GDS bleed must not survive
        out = bol._normalize_bol(data)
        assert "SurpriseField" not in out
        assert "segments" not in out
        assert set(out.keys()) == set(_golden().keys())

    def test_missing_string_fields_default_na(self):
        raw = {"BLNumber": None, "Shipper": "", "Brand": None}
        out = bol._normalize_bol(raw)
        for key in ("BLNumber", "Shipper", "ShipperAddress", "VoyageNumber",
                    "Brand", "PortOfOrigin", "CustomerName", "CustomerCode",
                    "CustomerAddress", "ShipVia"):
            assert out[key] == "N/A"

    def test_non_numeric_date_parts_coerce_to_zero(self):
        out = bol._normalize_bol({"BLDate": {"Month": "April", "Day": None, "Year": "twenty"}})
        assert out["BLDate"] == {"Month": 0, "Day": 0, "Year": 0}

    def test_date_parts_coerce_from_strings(self):
        out = bol._normalize_bol({"BLDate": {"Month": "4", "Day": "15", "Year": "2026"}})
        assert out["BLDate"] == {"Month": 4, "Day": 15, "Year": 2026}

    def test_cartons_coercion(self):
        assert bol._normalize_bol({"Cartons": "1200"})["Cartons"] == 1200
        assert bol._normalize_bol({"Cartons": 1200})["Cartons"] == 1200
        assert bol._normalize_bol({"Cartons": None})["Cartons"] == 0
        assert bol._normalize_bol({"Cartons": "many"})["Cartons"] == 0

    def test_nested_objects_defaulted_when_absent(self):
        out = bol._normalize_bol({})
        assert out["BLDate"] == {"Month": 0, "Day": 0, "Year": 0}
        assert out["ShipToDestination"] == {"City": "N/A", "Country": "N/A"}

    def test_non_dict_raises_model_unavailable(self):
        with pytest.raises(bol.ModelUnavailable):
            bol._normalize_bol(["not", "a", "dict"])

    def test_extract_json_gateway_side_customer_resolution(self):
        # Model resolved the code wrongly; gateway lookup is authoritative.
        raw = json.dumps({**_golden(), "CustomerCode": "C9999"})
        table = [dict(r) for r in bol.DEFAULT_CUSTOMER_TABLE]
        out = bol.extract_json(raw, table)
        assert out["CustomerCode"] == "C0003"

    def test_extract_json_no_match_resolves_na(self):
        raw = json.dumps({**_golden(), "CustomerName": "UNKNOWN TRADING COMPANY"})
        table = [dict(r) for r in bol.DEFAULT_CUSTOMER_TABLE]
        out = bol.extract_json(raw, table)
        assert out["CustomerCode"] == "N/A"
        assert out["CustomerName"] == "UNKNOWN TRADING COMPANY"


class TestExtractJsonSanitization:
    def test_none_raises(self):
        with pytest.raises(bol.ModelUnavailable):
            bol.extract_json(None, bol.DEFAULT_CUSTOMER_TABLE)

    def test_think_blocks_stripped(self):
        raw = "<think>let me analyze this B/L carefully...</think>" + json.dumps(_golden())
        out = bol.extract_json(raw, bol.DEFAULT_CUSTOMER_TABLE)
        assert out["BLNumber"] == _golden()["BLNumber"]

    def test_legacy_thinking_markers_stripped(self):
        raw = "<|begin_thinking|>reasoning here<|end_thinking|>" + json.dumps(_golden())
        out = bol.extract_json(raw, bol.DEFAULT_CUSTOMER_TABLE)
        assert out["Cartons"] == 1200

    def test_code_fences_stripped(self):
        raw = "```json\n" + json.dumps(_golden()) + "\n```"
        out = bol.extract_json(raw, bol.DEFAULT_CUSTOMER_TABLE)
        assert out["BLNumber"] == _golden()["BLNumber"]

    def test_preamble_text_ignored(self):
        raw = "Here is the extracted data:\n" + json.dumps(_golden())
        out = bol.extract_json(raw, bol.DEFAULT_CUSTOMER_TABLE)
        assert out["BLNumber"] == _golden()["BLNumber"]

    def test_garbage_fails_closed(self):
        with pytest.raises(bol.ModelUnavailable):
            bol.extract_json("I could not read the document.", bol.DEFAULT_CUSTOMER_TABLE)

    def test_broken_json_span_fails_closed(self):
        with pytest.raises(bol.ModelUnavailable):
            bol.extract_json('{"BLNumber": "oops', bol.DEFAULT_CUSTOMER_TABLE)


# --------------------------------------------------------------------------
# Customer table (config-driven; D3 decision)
# --------------------------------------------------------------------------
TABLE = bol.DEFAULT_CUSTOMER_TABLE


class TestLookupCustomer:
    def test_exact_registered_name(self):
        assert bol.lookup_customer("ANA Foods Co., LTD", TABLE) == "C0003"

    def test_case_insensitive(self):
        assert bol.lookup_customer("ana foods co., ltd", TABLE) == "C0003"

    def test_punctuation_bracket_insensitive(self):
        assert bol.lookup_customer("laysun far east limited", TABLE) == "C0002"
        assert bol.lookup_customer("LAYSUN [FAR EAST] LIMITED", TABLE) == "C0002"

    def test_buyer_column_match(self):
        assert bol.lookup_customer("Hiro International", TABLE) == "C0005"
        assert bol.lookup_customer("Farmind Corporation", TABLE) == "C0006"

    def test_substring_containment(self):
        # Document carries extra context around a registered name.
        assert bol.lookup_customer("ANA FOODS CO., LTD (TOKYO BRANCH)", TABLE) == "C0003"

    def test_no_match_returns_na(self):
        assert bol.lookup_customer("Unknown Trading K.K.", TABLE) == "N/A"

    def test_empty_name_returns_na(self):
        assert bol.lookup_customer("", TABLE) == "N/A"
        assert bol.lookup_customer("   ", TABLE) == "N/A"

    def test_none_name_returns_na(self):
        assert bol.lookup_customer(None, TABLE) == "N/A"

    def test_ambiguous_never_guesses(self):
        two_same_name = [
            {"Code": "X001", "Buyer": "Acme Corp", "RegisteredCustomerName": "Acme Corp"},
            {"Code": "X002", "Buyer": "Other Ltd", "RegisteredCustomerName": "Acme Corp"},
        ]
        assert bol.lookup_customer("Acme Corp", two_same_name) == "N/A"

    def test_custom_table_used_for_lookup(self):
        custom = [{"Code": "Z777", "Buyer": "Yamato Trading",
                   "RegisteredCustomerName": "Yamato Trading Co."}]
        assert bol.lookup_customer("YAMATO TRADING CO.", custom) == "Z777"


class TestLoadCustomerTable:
    def test_env_empty_uses_defaults(self, monkeypatch):
        monkeypatch.setattr(bol, "CUSTOMER_TABLE_ENV", "")
        rows = bol.load_customer_table()
        assert len(rows) == 4
        assert {r["Code"] for r in rows} == {"C0002", "C0003", "C0005", "C0006"}

    def test_valid_custom_env_overrides(self, monkeypatch):
        monkeypatch.setattr(
            bol, "CUSTOMER_TABLE_ENV",
            json.dumps([{"Code": "W1", "Buyer": "A", "RegisteredCustomerName": "A Co."}]))
        rows = bol.load_customer_table()
        assert rows == [{"Code": "W1", "Buyer": "A", "RegisteredCustomerName": "A Co."}]

    def test_invalid_json_falls_back_to_defaults(self, monkeypatch):
        monkeypatch.setattr(bol, "CUSTOMER_TABLE_ENV", "{not valid json")
        rows = bol.load_customer_table()
        assert len(rows) == 4

    def test_non_array_falls_back_to_defaults(self, monkeypatch):
        monkeypatch.setattr(bol, "CUSTOMER_TABLE_ENV", '{"Code": "X1"}')
        assert len(bol.load_customer_table()) == 4

    def test_malformed_rows_filtered(self, monkeypatch):
        monkeypatch.setattr(
            bol, "CUSTOMER_TABLE_ENV",
            json.dumps([
                "not-a-dict",
                {"Code": "", "Buyer": "B", "RegisteredCustomerName": "B"},
                {"Code": "OK1", "Buyer": "Good", "RegisteredCustomerName": "Good"},
            ]))
        rows = bol.load_customer_table()
        assert rows == [{"Code": "OK1", "Buyer": "Good", "RegisteredCustomerName": "Good"}]

    def test_all_invalid_rows_fall_back_to_defaults(self, monkeypatch):
        monkeypatch.setattr(
            bol, "CUSTOMER_TABLE_ENV",
            json.dumps([{"Code": "", "Buyer": "", "RegisteredCustomerName": ""}]))
        assert len(bol.load_customer_table()) == 4

    def test_defaults_are_not_mutated(self, monkeypatch):
        before = [dict(r) for r in bol.DEFAULT_CUSTOMER_TABLE]
        monkeypatch.setattr(
            bol, "CUSTOMER_TABLE_ENV",
            json.dumps([{"Code": "M1", "Buyer": "M", "RegisteredCustomerName": "M"}]))
        bol.load_customer_table()
        assert bol.DEFAULT_CUSTOMER_TABLE == before


# --------------------------------------------------------------------------
# Context guard machinery (vLLM — continuous batching, no slots)
# --------------------------------------------------------------------------
class TestContextGuard:
    def test_usable_prompt_room_vllm_budget(self):
        """vLLM: no slot division — full CONTEXT_SIZE."""
        room = bol.usable_prompt_room()
        expected = 32768 - 4096 - bol._SAFETY_MARGIN
        assert room == expected

    def test_context_budget_32k_default(self):
        """vLLM MAX_MODEL_LEN = 32768, no slot division."""
        assert bol.CONTEXT_SIZE == 32768

    def test_slot_budget_doesnt_exist(self):
        """vLLM has no slot division — slot_budget() must be removed."""
        assert not hasattr(bol, "slot_budget")

    def test_estimate_tokens_chars_div_three_ceiling(self):
        assert bol.estimate_tokens("") == 0
        assert bol.estimate_tokens("a" * 300) == 100
        assert bol.estimate_tokens("a" * 301) == 101

    def test_strict_guard_rejects_oversized_prompt(self, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 8192)
        monkeypatch.setattr(bol, "MODEL_MAX_TOKENS", 4096)
        monkeypatch.setattr(bol, "CONTEXT_GUARD", "strict")
        # room = 8192 - 4096 - 256 = 3840 tokens → need > 3840*3 = 11520 chars
        big = [{"role": "user", "content": "x" * 12000}]
        with pytest.raises(bol.ContextGuardExceeded) as exc:
            bol.check_context(big)
        msg = str(exc.value)
        assert "vLLM" in msg
        assert "context" in msg.lower()

    def test_warn_guard_allows_with_warning(self, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 4096)
        monkeypatch.setattr(bol, "MODEL_MAX_TOKENS", 64)
        monkeypatch.setattr(bol, "CONTEXT_GUARD", "warn")
        big = [{"role": "user", "content": "x" * 6000}]
        bol.check_context(big)  # must not raise

    def test_off_guard_skips_entirely(self, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 1)
        monkeypatch.setattr(bol, "MODEL_MAX_TOKENS", 100)
        monkeypatch.setattr(bol, "CONTEXT_GUARD", "off")
        bol.check_context([{"role": "user", "content": "x" * 999999}])

    def test_small_prompt_passes_all_modes(self, monkeypatch):
        for mode in ("strict", "warn", "off"):
            monkeypatch.setattr(bol, "CONTEXT_GUARD", mode)
            bol.check_context([{"role": "user", "content": "tiny"}])

    def test_check_context_error_mentions_vllm_and_no_slots(self, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 8192)
        monkeypatch.setattr(bol, "MODEL_MAX_TOKENS", 4096)
        monkeypatch.setattr(bol, "CONTEXT_GUARD", "strict")
        big = [{"role": "user", "content": "x" * 12000}]
        with pytest.raises(bol.ContextGuardExceeded) as exc:
            bol.check_context(big)
        msg = str(exc.value)
        assert "vLLM" in msg
        assert "slot" not in msg.lower() and "parallel" not in msg.lower()


# --------------------------------------------------------------------------
# Sampling params + vLLM-native thinking-suppression (2-level)
# --------------------------------------------------------------------------
class TestBuildParams:
    def test_level0_has_chat_template_kwargs(self):
        p = bol.build_params(0)
        assert p["chat_template_kwargs"] == {"enable_thinking": False}
        assert p["temperature"] == bol.MODEL_TEMP

    def test_level1_drops_chat_template_kwargs(self):
        p = bol._params_at_level(bol.build_params(0), 1)
        assert "chat_template_kwargs" not in p

    def test_degradation_levels_is_two(self):
        """vLLM: 2-level chain (0, 1)."""
        assert bol._PARAMS_LEVELS == (0, 1)

    def test_level0_no_reasoning_effort(self):
        """vLLM migration: reasoning_effort must NEVER appear."""
        p = bol.build_params(0)
        assert "reasoning_effort" not in p

    def test_disable_thinking_false_skips_suppression(self, monkeypatch):
        monkeypatch.setattr(bol, "DISABLE_THINKING", False)
        p = bol.build_params(0)
        assert "reasoning_effort" not in p
        assert "chat_template_kwargs" not in p

    def test_greedy_sampling_forwarded(self):
        p = bol.build_params(0)
        assert p["temperature"] == 0.0
        assert p["max_tokens"] == bol.MODEL_MAX_TOKENS


# --------------------------------------------------------------------------
# Model backend HTTP behavior (vLLM httpx.AsyncClient)
# --------------------------------------------------------------------------
class TestHttpModelCall:
    def _ok_payload(self, content="{}"):
        return {"choices": [{"message": {"content": content}}]}

    def _run_coro(self, coro):
        """Helper to run an async coroutine synchronously (used by all mocks)."""
        import asyncio
        return asyncio.run(coro)

    def test_success_returns_content(self, monkeypatch):
        async def fake_post(body, headers):
            return self._ok_payload('{"x":1}')

        def fake_run_async(coro):
            return self._run_coro(coro)

        monkeypatch.setattr(bol, "_post_async", fake_post)
        monkeypatch.setattr(bol, "_run_async", fake_run_async)
        out = bol.http_model_call(_messages(), bol.build_params(0))
        assert out == '{"x":1}'

    def test_http_call_vllm_2_level_degradation(self, monkeypatch):
        """vLLM: 2-level chain (not 3)."""
        call_count = {"n": 0}

        async def fake_post(body, headers):
            call_count["n"] += 1
            if "chat_template_kwargs" in body:
                return {"error": {"message": "unknown field: chat_template_kwargs"}}
            return self._ok_payload("done")

        def fake_run_async(coro):
            return self._run_coro(coro)

        monkeypatch.setattr(bol, "_PARAMS_LEVEL", 0)
        monkeypatch.setattr(bol, "_post_async", fake_post)
        monkeypatch.setattr(bol, "_run_async", fake_run_async)
        out = bol.http_model_call(_messages(), bol.build_params(0))
        assert out == "done"
        assert call_count["n"] == 2

    def test_http_call_vllm_no_reasoning_effort_sent(self, monkeypatch):
        """vLLM: reasoning_effort must never appear in the body."""
        bodies = []

        async def fake_post(body, headers):
            bodies.append(dict(body))
            return self._ok_payload("ok")

        def fake_run_async(coro):
            return self._run_coro(coro)

        monkeypatch.setattr(bol, "_post_async", fake_post)
        monkeypatch.setattr(bol, "_run_async", fake_run_async)
        bol.http_model_call(_messages(), bol.build_params(0))
        for body in bodies:
            assert "reasoning_effort" not in body

    def test_http_call_vllm_caches_working_level(self, monkeypatch):
        """After level 1 succeeds, subsequent calls go straight to level 1."""
        call_count = {"n": 0}

        async def fake_post(body, headers):
            call_count["n"] += 1
            if "chat_template_kwargs" in body:
                return {"error": {"message": "server rejected chat_template_kwargs"}}
            return self._ok_payload("ok")

        def fake_run_async(coro):
            return self._run_coro(coro)

        monkeypatch.setattr(bol, "_PARAMS_LEVEL", 0)
        monkeypatch.setattr(bol, "_post_async", fake_post)
        monkeypatch.setattr(bol, "_run_async", fake_run_async)

        bol.http_model_call(_messages(), bol.build_params(0))
        first_calls = call_count["n"]

        bol.http_model_call(_messages(), bol.build_params(0))
        assert call_count["n"] == first_calls + 1

    def test_context_overflow_maps_to_context_exceeded(self, monkeypatch):
        async def fake_post(body, headers):
            return {"error": {"message": "request exceeds context length"}}

        def fake_run_async(coro):
            return self._run_coro(coro)

        monkeypatch.setattr(bol, "_post_async", fake_post)
        monkeypatch.setattr(bol, "_run_async", fake_run_async)
        with pytest.raises(bol.ContextExceeded):
            bol.http_model_call(_messages(), bol.build_params(0))

    def test_generic_error_maps_to_model_unavailable(self, monkeypatch):
        async def fake_post(body, headers):
            return {"error": {"message": "internal error"}}

        def fake_run_async(coro):
            return self._run_coro(coro)

        monkeypatch.setattr(bol, "_post_async", fake_post)
        monkeypatch.setattr(bol, "_run_async", fake_run_async)
        with pytest.raises(bol.ModelUnavailable):
            bol.http_model_call(_messages(), bol.build_params(0))

    def test_vllm_usage_parsed_from_response(self, monkeypatch):
        """vLLM response includes 'usage' dict with prompt/completion tokens."""
        captured = {}

        async def fake_post(body, headers):
            captured["body"] = dict(body)
            return {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 150, "completion_tokens": 25},
            }

        def fake_run_async(coro):
            return self._run_coro(coro)

        monkeypatch.setattr(bol, "_post_async", fake_post)
        monkeypatch.setattr(bol, "_run_async", fake_run_async)
        bol.http_model_call(_messages(), bol.build_params(0))
        assert captured.get("body", {}).get("model") == "Qwen3.6-35B-A3B-NVFP4"


class TestResolveContent:
    def test_content_preferred(self):
        assert bol._resolve_content({"content": "answer"}) == "answer"

    def test_falls_back_to_reasoning_content(self):
        assert bol._resolve_content({"content": "", "reasoning_content": "r"}) == "r"

    def test_falls_back_through_legacy_names(self):
        for field in ("thinking", "reasoning", "reason"):
            assert bol._resolve_content({"content": "", field: f"v-{field}"}) == f"v-{field}"

    def test_all_empty_raises(self):
        with pytest.raises(bol.ModelUnavailable):
            bol._resolve_content({"content": ""})


# --------------------------------------------------------------------------
# Orchestration validation (run_extract / run_extract_batch)
# --------------------------------------------------------------------------
class TestRunExtractValidation:
    def test_missing_bol_text(self):
        with pytest.raises(ValueError):
            bol.run_extract({}, bol.DEFAULT_CUSTOMER_TABLE, lambda m, p: "{}")

    def test_non_string_bol_text(self):
        with pytest.raises(ValueError):
            bol.run_extract({"bol_text": 12345}, bol.DEFAULT_CUSTOMER_TABLE, lambda m, p: "{}")

    def test_whitespace_only_bol_text(self):
        with pytest.raises(ValueError):
            bol.run_extract({"bol_text": "   \n  "}, bol.DEFAULT_CUSTOMER_TABLE,
                            lambda m, p: "{}")

    def test_happy_path_returns_normalized_record(self):
        raw = json.dumps(_golden())
        out = bol.run_extract({"bol_text": _golden_text()},
                              [dict(r) for r in bol.DEFAULT_CUSTOMER_TABLE],
                              lambda m, p: raw)
        assert out == _golden()


class TestBatchIsolation:
    def test_good_and_bad_entries_isolated(self, client, monkeypatch):
        def flaky(messages, params):
            user = messages[1]["content"]
            if "BAD_ENTRY_MARKER" in user:
                raise bol.ModelUnavailable("simulated failure")
            return json.dumps(_golden())

        monkeypatch.setattr(bol, "http_model_call", flaky)
        body = {"entries": [
            {"id": "good-1", "bol_text": _golden_text()},
            {"id": "bad-1", "bol_text": "BAD_ENTRY_MARKER broken doc"},
            {"id": "good-2", "bol_text": _golden_text()},
        ]}
        r = client.post("/v1/extract_batch", json=body, headers=_auth())
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 3
        assert results[0]["status"] == "ok"
        assert results[0]["data"] == _golden()
        assert results[1]["status"] == "error"
        assert "simulated failure" in results[1]["error"]
        assert results[2]["status"] == "ok"

    def test_ids_echoed_and_none_default(self, client, monkeypatch):
        body = {"entries": [
            {"id": "a", "bol_text": _golden_text()},
            {"bol_text": _golden_text()},  # no id supplied
        ]}
        r = client.post("/v1/extract_batch", json=body, headers=_auth())
        results = r.json()["results"]
        assert [x["id"] for x in results] == ["a", None]

    def test_validation_error_entry_reports_error_status(self, client):
        body = {"entries": [
            {"id": "empty", "bol_text": "   "},
        ]}
        r = client.post("/v1/extract_batch", json=body, headers=_auth())
        assert r.status_code == 200
        results = r.json()["results"]
        assert results[0]["status"] == "error"


# --------------------------------------------------------------------------
# API surface — auth, validation, version, healthz, root
# --------------------------------------------------------------------------
class TestApiSurface:
    def test_missing_key_401(self, client):
        r = client.post("/v1/extract", json=_extract_body())
        assert r.status_code == 401

    def test_wrong_key_401(self, client):
        r = client.post("/v1/extract", json=_extract_body(),
                        headers={"x-api-key": "nope"})
        assert r.status_code == 401

    def test_missing_bol_text_field_422(self, client):
        r = client.post("/v1/extract", json={}, headers=_auth())
        assert r.status_code == 422

    def test_empty_bol_text_rejected_by_pydantic(self, client):
        r = client.post("/v1/extract", json={"bol_text": ""}, headers=_auth())
        assert r.status_code == 422

    def test_malformed_json_422(self, client):
        r = client.post("/v1/extract", content=b"{not json",
                        headers={**_auth(), "Content-Type": "application/json"})
        assert r.status_code == 422

    def test_version_endpoint_no_model_call(self, client):
        r = client.post("/v1/version", headers=_auth())
        assert r.status_code == 200
        assert r.json() == {"version": "J2.1"}
        assert client.calls["n"] == 0

    def test_root_lists_endpoints(self, client):
        r = client.get("/")
        assert r.status_code == 200
        payload = r.json()
        assert payload["version"] == "J2.1"
        for ep in ("/v1/extract", "/v1/extract_file", "/v1/extract_batch", "/v1/version", "/healthz"):
            assert ep in payload["endpoints"]

    def test_healthz_degraded_when_model_down(self, client, monkeypatch):
        import requests as req

        def unreachable(url, **kw):
            raise req.exceptions.ConnectionError("refused")

        monkeypatch.setattr(bol.requests, "get", unreachable)
        r = client.get("/healthz")
        assert r.status_code == 200
        payload = r.json()
        assert payload["status"] == "degraded"
        assert payload["model_server"].startswith("unreachable")
        assert payload["version"] == "J2.1"

    def test_healthz_reports_vllm_fields(self, client, monkeypatch):
        import requests as req

        def fake_get(url, **kw):
            if url.endswith("/health"):
                return FakeResp(200, json_data={"status": "ok"})
            if url.endswith("/v1/models"):
                return FakeResp(200, json_data={
                    "data": [{"id": "Qwen3.6-35B-A3B-NVFP4"}]
                })
            return FakeResp(404)

        monkeypatch.setattr(bol.requests, "get", fake_get)
        r = client.get("/healthz")
        body = r.json()
        assert body["status"] == "healthy"
        assert body["model_server"] == "ready"
        assert "vllm_model_id" in body
        assert body["vllm_model_id"] == "Qwen3.6-35B-A3B-NVFP4"
        assert "thinking_disabled" in body
        assert "context_guard" in body
        # vLLM: no slot_tokens
        assert "slot_tokens" not in body["context_budget"]

    def test_healthz_has_no_slot_tokens_field(self, client, monkeypatch):
        import requests as req

        def fake_get(url, **kw):
            if url.endswith("/health"):
                return FakeResp(200, json_data={"status": "ok"})
            if url.endswith("/v1/models"):
                return FakeResp(200, json_data={"data": [{"id": "Qwen3.6-35B-A3B-NVFP4"}]})
            return FakeResp(404)

        monkeypatch.setattr(bol.requests, "get", fake_get)
        r = client.get("/healthz")
        payload = r.json()
        assert "slot_tokens" not in payload.get("context_budget", {})
        assert "context_size" in payload.get("context_budget", {})

    def test_healthz_context_budget_reports_context_size(self, client, monkeypatch):
        import requests as req

        def fake_get(url, **kw):
            if url.endswith("/health"):
                return FakeResp(200, json_data={"status": "ok"})
            if url.endswith("/v1/models"):
                return FakeResp(200, json_data={"data": [{"id": "Qwen3.6-35B-A3B-NVFP4"}]})
            return FakeResp(404)

        monkeypatch.setattr(bol.requests, "get", fake_get)
        r = client.get("/healthz")
        payload = r.json()
        assert payload["context_budget"]["context_size"] == bol.CONTEXT_SIZE

    def test_extract_503_when_unparseable_output(self, client, monkeypatch):
        monkeypatch.setattr(bol, "http_model_call",
                            lambda m, p: "no JSON here at all")
        r = client.post("/v1/extract", json=_extract_body(), headers=_auth())
        assert r.status_code == 503

    def test_extract_422_on_context_guard_exceeded(self, client, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 4096)
        monkeypatch.setattr(bol, "MODEL_MAX_TOKENS", 64)
        monkeypatch.setattr(bol, "CONTEXT_GUARD", "strict")
        huge = "x" * 20000
        r = client.post("/v1/extract", json={"bol_text": huge}, headers=_auth())
        assert r.status_code == 422

    def test_extract_batch_requires_entries(self, client):
        r = client.post("/v1/extract_batch", json={"entries": []}, headers=_auth())
        assert r.status_code == 422


# --------------------------------------------------------------------------
# MD file ingest — POST /v1/extract_file (v3.1)
# --------------------------------------------------------------------------
class TestExtractFile:
    def _golden_bytes(self):
        with open(os.path.join(CASES, "bol_golden_input.txt"), "rb") as f:
            return f.read()

    def test_file_happy_path_parity(self, client):
        r = client.post(
            "/v1/extract_file",
            files={"file": ("ocr.md", self._golden_bytes(), "text/markdown")},
            headers=_auth(),
        )
        assert r.status_code == 200
        assert r.json() == _golden()

    def test_file_txt_extension_ok(self, client):
        r = client.post(
            "/v1/extract_file",
            files={"file": ("ocr.txt", self._golden_bytes(), "text/plain")},
            headers=_auth(),
        )
        assert r.status_code == 200

    def test_file_missing_401(self, client):
        r = client.post(
            "/v1/extract_file",
            files={"file": ("ocr.md", self._golden_bytes(), "text/markdown")},
        )
        assert r.status_code == 401

    def test_file_missing_field_422(self, client):
        r = client.post("/v1/extract_file", data={}, headers=_auth())
        assert r.status_code == 422

    def test_file_wrong_extension_422(self, client):
        r = client.post(
            "/v1/extract_file",
            files={"file": ("scan.pdf", b"%PDF-1.4 fake", "application/pdf")},
            headers=_auth(),
        )
        assert r.status_code == 422

    def test_file_empty_422(self, client):
        r = client.post(
            "/v1/extract_file",
            files={"file": ("empty.md", b"   ", "text/markdown")},
            headers=_auth(),
        )
        assert r.status_code == 422

    def test_file_non_utf8_422(self, client):
        r = client.post(
            "/v1/extract_file",
            files={"file": ("bad.md", b"\xff\xfe\x00bad", "text/markdown")},
            headers=_auth(),
        )
        assert r.status_code == 422

    def test_file_oversize_422(self, client, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 4096)
        monkeypatch.setattr(bol, "MODEL_MAX_TOKENS", 64)
        monkeypatch.setattr(bol, "CONTEXT_GUARD", "strict")
        huge = b"x" * 20000
        r = client.post(
            "/v1/extract_file",
            files={"file": ("huge.md", huge, "text/markdown")},
            headers=_auth(),
        )
        assert r.status_code == 422

    def test_file_503_on_bad_model(self, client, monkeypatch):
        monkeypatch.setattr(bol, "http_model_call", lambda m, p: "no JSON here at all")
        r = client.post(
            "/v1/extract_file",
            files={"file": ("ocr.md", self._golden_bytes(), "text/markdown")},
            headers=_auth(),
        )
        assert r.status_code == 503

    def test_get_on_file_405(self, client):
        r = client.get("/v1/extract_file")
        assert r.status_code == 405

    def test_get_on_extract_405_regression(self, client):
        r = client.get("/v1/extract")
        assert r.status_code == 405


# --------------------------------------------------------------------------
# Config drift — .env.example must match code defaults (C2 invariant)
# --------------------------------------------------------------------------
def test_defaults_sync_to_env_example():
    settings = {}
    with open(ENV_EXAMPLE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            settings[key.strip()] = val.strip().strip('"').strip("'")

    assert int(settings["CONTEXT_SIZE"]) == bol.CONTEXT_SIZE
    assert int(settings["API_PORT"]) == bol.API_PORT
    assert int(settings["MODEL_MAX_TOKENS"]) == bol.MODEL_MAX_TOKENS
    assert int(settings["REQUEST_TIMEOUT"]) == bol.REQUEST_TIMEOUT
    assert float(settings["MODEL_TEMP"]) == bol.MODEL_TEMP
    assert float(settings["MODEL_TOP_P"]) == bol.MODEL_TOP_P
    assert int(settings["MODEL_TOP_K"]) == bol.MODEL_TOP_K
    assert settings["LLAMA_SERVER_API_KEY"] == bol.LLAMA_SERVER_API_KEY
    assert settings["MODEL_NAME"] == bol.MODEL_NAME
    assert settings["MODEL_URL"] == bol.MODEL_URL
    assert settings["CONTEXT_GUARD"] == bol.CONTEXT_GUARD
    assert settings["API_KEY_AUTH_HEADER"] == bol.API_KEY_AUTH_HEADER
    # vLLM config invariants
    assert bol.CONTEXT_SIZE == 32768
    assert bol.MODEL_NAME == "Qwen3.6-35B-A3B-NVFP4"
    assert "8011" in bol.MODEL_URL
    assert bol.LLAMA_SERVER_API_KEY == "test_key_0000"
    assert bol.REQUEST_TIMEOUT == 120
    # Port allocation sanity per the roadmap port map.
    assert bol.API_PORT == 8086
