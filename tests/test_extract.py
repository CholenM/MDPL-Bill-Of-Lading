"""
Unit + integration tests for the AI MDPL Bill of Lading Extractor (v2.1).

These run on the Windows dev machine with a STUB model backend (no GPU, no
llama.cpp needed). The app's http_model_call is monkeypatched so the full
FastAPI stack can be exercised.

Run:  pytest -v
"""

import os
import json
import pytest
from fastapi.testclient import TestClient
import requests

import bol_service as bol

# Matches DEFAULT_KEYS used when no .env is present.
VALID_KEY = "bol_key_0000"
ENV_EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.example")
CASES = os.path.join(os.path.dirname(__file__), "cases")


class FakeResp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code, text="", payload=None):
        self.status_code = status_code
        self._text = text
        self._payload = payload or {}

    @property
    def text(self):
        return self._text

    def json(self):
        return self._payload


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
# Context guard machinery
# --------------------------------------------------------------------------
class TestContextGuard:
    def test_slot_budget_math(self, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 65536)
        monkeypatch.setattr(bol, "MODEL_PARALLEL", 4)
        assert bol.slot_budget() == 16384

    def test_slot_budget_minimum_one(self, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 1)
        monkeypatch.setattr(bol, "MODEL_PARALLEL", 100)
        assert bol.slot_budget() >= 1

    def test_usable_prompt_room(self, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 65536)
        monkeypatch.setattr(bol, "MODEL_PARALLEL", 4)
        monkeypatch.setattr(bol, "MODEL_MAX_TOKENS", 4096)
        assert bol.usable_prompt_room() == 16384 - 4096 - bol._SAFETY_MARGIN

    def test_estimate_tokens_chars_div_three_ceiling(self):
        assert bol.estimate_tokens("") == 0
        assert bol.estimate_tokens("a" * 300) == 100
        assert bol.estimate_tokens("a" * 301) == 101

    def test_strict_guard_rejects_oversized_prompt(self, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 4096)
        monkeypatch.setattr(bol, "MODEL_PARALLEL", 4)   # slot=1024
        monkeypatch.setattr(bol, "MODEL_MAX_TOKENS", 64)
        monkeypatch.setattr(bol, "CONTEXT_GUARD", "strict")
        big = [{"role": "user", "content": "x" * 6000}]
        with pytest.raises(bol.ContextGuardExceeded) as exc:
            bol.check_context(big)
        msg = str(exc.value)
        assert "re-provision" in msg or "slot budget" in msg

    def test_warn_guard_allows_with_warning(self, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 4096)
        monkeypatch.setattr(bol, "MODEL_PARALLEL", 4)
        monkeypatch.setattr(bol, "MODEL_MAX_TOKENS", 64)
        monkeypatch.setattr(bol, "CONTEXT_GUARD", "warn")
        big = [{"role": "user", "content": "x" * 6000}]
        bol.check_context(big)  # must not raise

    def test_off_guard_skips_entirely(self, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 1)
        monkeypatch.setattr(bol, "MODEL_PARALLEL", 100)
        monkeypatch.setattr(bol, "CONTEXT_GUARD", "off")
        bol.check_context([{"role": "user", "content": "x" * 999999}])

    def test_small_prompt_passes_all_modes(self, monkeypatch):
        for mode in ("strict", "warn", "off"):
            monkeypatch.setattr(bol, "CONTEXT_GUARD", mode)
            bol.check_context([{"role": "user", "content": "tiny"}])


# --------------------------------------------------------------------------
# Sampling params + thinking-suppression degradation chain
# --------------------------------------------------------------------------
class TestBuildParams:
    def test_level0_includes_both_suppression_fields(self):
        p = bol.build_params(0)
        assert p["reasoning_effort"] == 0
        assert p["chat_template_kwargs"] == {"enable_thinking": False}
        assert p["temperature"] == bol.MODEL_TEMP

    def test_level1_drops_chat_template_kwargs_only(self):
        p = bol._params_at_level(bol.build_params(0), 1)
        assert "chat_template_kwargs" not in p
        assert p["reasoning_effort"] == 0

    def test_level2_drops_reasoning_effort_too(self):
        p = bol._params_at_level(bol.build_params(0), 2)
        assert "chat_template_kwargs" not in p
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
# Model backend HTTP behavior (FakeResp-based)
# --------------------------------------------------------------------------
class TestHttpModelCall:
    def _ok_payload(self, content="{}"):
        return {"choices": [{"message": {"content": content}}]}

    def test_success_returns_content(self, monkeypatch):
        monkeypatch.setattr(bol.requests, "post",
                            lambda *a, **k: FakeResp(200, payload=self._ok_payload('{"x":1}')))
        out = bol.http_model_call(_messages(), bol.build_params(0))
        assert out == '{"x":1}'

    def test_degradation_chain_drops_rejected_params(self, monkeypatch):
        bodies = []

        def fake_post(url, json=None, headers=None, timeout=None):
            bodies.append(dict(json))
            if "chat_template_kwargs" in json:
                return FakeResp(400, text="unknown field: chat_template_kwargs")
            return FakeResp(200, payload=self._ok_payload("done"))

        monkeypatch.setattr(bol, "_PARAMS_LEVEL", 0)
        monkeypatch.setattr(bol.requests, "post", fake_post)
        out = bol.http_model_call(_messages(), bol.build_params(0))
        assert out == "done"
        assert len(bodies) == 2
        assert "chat_template_kwargs" not in bodies[-1]
        assert "reasoning_effort" in bodies[-1]

    def test_degradation_caches_working_level(self, monkeypatch):
        calls = {"n": 0}

        def fake_post(url, json=None, headers=None, timeout=None):
            calls["n"] += 1
            if "chat_template_kwargs" in json:
                return FakeResp(400, text="server rejected chat_template_kwargs")
            return FakeResp(200, payload=self._ok_payload("ok"))

        monkeypatch.setattr(bol, "_PARAMS_LEVEL", 0)
        monkeypatch.setattr(bol.requests, "post", fake_post)
        bol.http_model_call(_messages(), bol.build_params(0))
        first_calls = calls["n"]
        # Second call should skip straight to the cached level.
        bol.http_model_call(_messages(), bol.build_params(0))
        assert calls["n"] == first_calls + 1

    def test_context_overflow_maps_to_context_exceeded(self, monkeypatch):
        monkeypatch.setattr(bol.requests, "post",
                            lambda *a, **k: FakeResp(400, text="request exceeds context length"))
        with pytest.raises(bol.ContextExceeded):
            bol.http_model_call(_messages(), bol.build_params(0))

    def test_generic_error_maps_to_model_unavailable(self, monkeypatch):
        monkeypatch.setattr(bol.requests, "post",
                            lambda *a, **k: FakeResp(500, text="internal error"))
        with pytest.raises(bol.ModelUnavailable):
            bol.http_model_call(_messages(), bol.build_params(0))

    def test_connection_error_maps_to_model_unavailable(self, monkeypatch):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(bol.requests, "post", boom)
        with pytest.raises(bol.ModelUnavailable):
            bol.http_model_call(_messages(), bol.build_params(0))


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
        for ep in ("/v1/extract", "/v1/extract_batch", "/v1/version", "/healthz"):
            assert ep in payload["endpoints"]

    def test_healthz_degraded_when_model_down(self, client, monkeypatch):
        def unreachable(url, **kw):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(bol.requests, "get", unreachable)
        r = client.get("/healthz")
        assert r.status_code == 200
        payload = r.json()
        assert payload["status"] == "degraded"
        assert payload["model_server"].startswith("unreachable")
        assert payload["version"] == "J2.1"

    def test_healthz_ok_with_context_crosscheck(self, client, monkeypatch):
        def fake_get(url, **kw):
            if url.endswith("/health"):
                return FakeResp(200, payload={"n_ctx": 65536})
            if url.endswith("/props"):
                return FakeResp(200, payload={
                    "default_generation_settings": {"n_ctx": bol.slot_budget()}})
            return FakeResp(404)

        monkeypatch.setattr(bol.requests, "get", fake_get)
        r = client.get("/healthz")
        payload = r.json()
        assert payload["status"] == "healthy"
        assert payload["context_budget"]["check"] == "match"

    def test_healthz_context_mismatch_reported(self, client, monkeypatch):
        def fake_get(url, **kw):
            if url.endswith("/health"):
                return FakeResp(200, payload={"n_ctx": 65536})
            if url.endswith("/props"):
                return FakeResp(200, payload={
                    "default_generation_settings": {"n_ctx": bol.slot_budget() + 999}})
            return FakeResp(404)

        monkeypatch.setattr(bol.requests, "get", fake_get)
        r = client.get("/healthz")
        assert r.json()["context_budget"]["check"] == "mismatch"

    def test_extract_503_when_unparseable_output(self, client, monkeypatch):
        monkeypatch.setattr(bol, "http_model_call",
                            lambda m, p: "no JSON here at all")
        r = client.post("/v1/extract", json=_extract_body(), headers=_auth())
        assert r.status_code == 503

    def test_extract_422_on_context_guard_exceeded(self, client, monkeypatch):
        monkeypatch.setattr(bol, "CONTEXT_SIZE", 4096)      # slot=1024
        monkeypatch.setattr(bol, "MODEL_PARALLEL", 4)
        monkeypatch.setattr(bol, "MODEL_MAX_TOKENS", 64)
        monkeypatch.setattr(bol, "CONTEXT_GUARD", "strict")
        huge = "x" * 20000
        r = client.post("/v1/extract", json={"bol_text": huge}, headers=_auth())
        assert r.status_code == 422

    def test_extract_batch_requires_entries(self, client):
        r = client.post("/v1/extract_batch", json={"entries": []}, headers=_auth())
        assert r.status_code == 422


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
    assert int(settings["MODEL_PARALLEL"]) == bol.MODEL_PARALLEL
    assert int(settings["API_PORT"]) == bol.API_PORT
    assert int(settings["MODEL_MAX_TOKENS"]) == bol.MODEL_MAX_TOKENS
    assert int(settings["REQUEST_TIMEOUT"]) == bol.REQUEST_TIMEOUT
    assert float(settings["MODEL_TEMP"]) == bol.MODEL_TEMP
    assert float(settings["MODEL_TOP_P"]) == bol.MODEL_TOP_P
    assert int(settings["MODEL_TOP_K"]) == bol.MODEL_TOP_K
    assert settings["LLAMA_SERVER_API_KEY"] == bol.LLAMA_SERVER_API_KEY
    assert settings["MODEL_NAME"] == bol.MODEL_NAME
    assert settings["CONTEXT_GUARD"] == bol.CONTEXT_GUARD
    assert settings["API_KEY_AUTH_HEADER"] == bol.API_KEY_AUTH_HEADER
    # Port allocation sanity per the roadmap port map.
    assert bol.API_PORT == 8086
