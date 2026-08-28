# MDPL Bill of Lading Extractor — v2.1 to v3 vLLM Migration Roadmap

> Source of truth for the Builder. This document describes the migration from
> llama.cpp (`:8006`, Qwen3.8-27B) to the shared vLLM server
> (`:8011`, Qwen3.6-35B-A3B-NVFP4), following the proven pattern from
> Proof-Reader, GDS Extraction, and QA-Manager.

---

## 1. Strategic Design

### 1.1 Objectives

Migrate the MDPL Bill of Lading Extractor FastAPI gateway from a synchronous
`requests.post()` call to llama.cpp's OpenAI-compatible API (`:8006`) to an
asynchronous `httpx.AsyncClient` call to the shared vLLM server (`:8011`).

The business logic (prompt building, customer lookup, JSON extraction,
normalization, context guard, batch orchestration, API routes) remains
**untouched** — only the model backend layer changes.

### 1.2 Architecture (Before → After)

```
BEFORE (llama.cpp):
  Client → FastAPI gateway (:8086) → requests.post() → llama-server (:8006)
  Qwen3.8-27B · ctx=65536 · parallel=4 · slot=16384

AFTER (vLLM):
  Client → FastAPI gateway (:8086) → httpx.AsyncClient → vLLM (:8011)
  Qwen3.6-35B-A3B-NVFP4 · MAX_MODEL_LEN=32768 · continuous batching · budget=32768
```

### 1.3 v3 Change Summary

| Aspect | Before (v2 / llama.cpp) | After (v3 / vLLM) |
|---|---|---|
| HTTP client | `requests.post()` (sync blocking) | `httpx.AsyncClient` (async) + `ThreadPoolExecutor` fallback for `model_call` |
| Model server | `:8006` (Qwen3.8-27B) | `:8011` (Qwen3.6-35B-A3B-NVFP4) |
| Context size | `CONTEXT_SIZE=65536`, `MODEL_PARALLEL=4` → 16384/slot | `CONTEXT_SIZE=32768`, continuous batching → full 32768 budget |
| Context guard error | References "slot budget" / `CONTEXT_SIZE // MODEL_PARALLEL` | References "vLLM context budget" (no slots) |
| Thinking suppression | 3-level chain: `reasoning_effort` → `chat_template_kwargs` → nothing | 2-level: `chat_template_kwargs` → nothing |
| Degradation levels | `_PARAMS_LEVELS = (0, 1, 2)` | `_PARAMS_LEVELS = (0, 1)` |
| `MODEL_PARALLEL` | Defined, used in slot budget math | **Removed** (not needed for continuous batching) |
| Health endpoint | `/health` + `/props` (llama.cpp-specific) | `/health` + `/v1/models` (vLLM OpenAI-compatible) |
| Timeout | `300s` | `120s` (vLLM is faster) |
| API key | `sk-internal-proofreader` | `test_key_0000` |
| Degradation loop | `while level <= 2` | `while level <= 1` |
| Server overflow error | `ContextExceeded` (503, from response body analysis) | `ContextExceeded` (503, from vLLM 400 response body analysis) |

### 1.4 Constraints (from Interrogation)

- **Context budget: 16k** — The vLLM server runs at `MAX_MODEL_LEN=32768`.
  For this project, the client-side guard will use `CONTEXT_SIZE=32768`
  (matching the server limit) but the *business allocation* is 16k per the
  Toby/GDMS context plan. The guard ensures no single prompt exceeds the
  server's 32k window, preventing vLLM strain during cross-system concurrency.
- **`MODEL_PARALLEL` removed** — vLLM uses continuous batching; no slot division.
- **2-level degradation only** — vLLM-native `chat_template_kwargs` with
  single-level retry (no `reasoning_effort`).
- **Strict client-side rejection** — Over-budget prompts return HTTP 422
  *before* any vLLM call, protecting shared server capacity.
- **120s timeout** — vLLM is faster than llama.cpp.
- **API key: `test_key_0000`** — Matches the shared vLLM server's auth token.
- **Port remains `8086`** — No change for ease of access.
- **Model name: `Qwen3.6-35B-A3B-NVFP4`** — The vLLM-served model.

### 1.5 Edge Cases & Failure Modes

| Scenario | Handling |
|---|---|
| vLLM server unreachable / timeout | `ModelUnavailable` → HTTP 503 |
| Prompt exceeds vLLM 32k context window | `ContextExceeded` → HTTP 503 (server-side, from response body) |
| Prompt exceeds client-side guard budget | `ContextGuardExceeded` → HTTP 422 (before any network call) |
| vLLM rejects `chat_template_kwargs` | Retry at level 1 (without the param); cached for subsequent calls |
| vLLM returns empty content | `ModelUnavailable` → HTTP 503 |
| Batch entry fails | Per-entry error recorded; overall 200 with `"status": "error"` |
| vLLM not ready (JIT boot) | Pre-flight waits up to 300s (vLLM needs up to 15min for FlashInfer JIT) |
| Context budget mismatch detection | `/healthz` reports `context_budget` from vLLM `--max-model-len` |

---

## 2. The Execution Roadmap (Task List)

### Phase A: Configuration & Dependencies

- [ ] **A1.** Update `.env.example`: change `MODEL_URL` to `:8011`, `MODEL_NAME` to
      `Qwen3.6-35B-A3B-NVFP4`, `CONTEXT_SIZE` to `32768`, remove `MODEL_PARALLEL`,
      change `LLAMA_SERVER_API_KEY` to `test_key_0000`, `REQUEST_TIMEOUT` to `120`.
      Update all comments to reflect vLLM, not llama.cpp.

- [ ] **A2.** Update `requirements.txt`: add `httpx>=0.27` to runtime deps (it's
      currently only in dev). Keep `requests>=2.31` for the `/healthz` and
      pre-flight probes (will be cleaned up later if needed).

### Phase B: Core Service Rewrite (`bol_service.py`)

- [ ] **B1.** Update module docstring (lines 1-34): replace llama.cpp references
      with vLLM, update architecture diagram, update model name to
      `Qwen3.6-35B-A3B-NVFP4`.

- [ ] **B2.** Configuration block (lines 58-96):
      - Change `CONTEXT_SIZE` default to `"32768"`.
      - **Remove `MODEL_PARALLEL` entirely** (delete lines 61 and all references).
      - Change `MODEL_URL` default to `"http://127.0.0.1:8011/v1/chat/completions"`.
      - Change `MODEL_NAME` default to `"Qwen3.6-35B-A3B-NVFP4"`.
      - Change `LLAMA_SERVER_API_KEY` default to `"test_key_0000"`.
      - Change `REQUEST_TIMEOUT` default to `"120"`.
      - Rename variable `LLAMA_SERVER_API_KEY` → `VLLM_API_KEY` (update all refs).

- [ ] **B3.** Degradation levels (lines 97-102):
      - Change `_PARAMS_LEVELS = (0, 1, 2)` → `_PARAMS_LEVELS = (0, 1)`.
      - Update the inline comment to describe vLLM-native 2-level chain.

- [ ] **B4.** `_base_params()` function (lines 352-362): **no change needed** —
      the sampling params dict is vLLM-compatible as-is.

- [ ] **B5.** `build_params()` function (lines 365-378):
      - **Remove `reasoning_effort`** — vLLM doesn't use this parameter.
      - Keep `chat_template_kwargs: {enable_thinking: False}` only (level 0).
      - Update docstring to say "vLLM-native, 2-level degradation."

- [ ] **B6.** `_params_at_level()` function (lines 381-388):
      - **Remove level 2 logic** (`out.pop("reasoning_effort", None)`).
      - Keep only level 1: `out.pop("chat_template_kwargs", None)`.

- [ ] **B7.** Context guard functions (lines 391-445):
      - Remove `slot_budget()` function entirely (no slot division in vLLM).
      - Rewrite `usable_prompt_room()` to use `CONTEXT_SIZE` directly
        (no division by parallel).
      - Rewrite `check_context()` error message: remove "slot budget" /
        `CONTEXT_SIZE // MODEL_PARALLEL` references. Replace with
        "vLLM context budget" / "vLLM context window."
      - Update guard to check against 32k (the vLLM server limit), not a
        per-slot budget.

- [ ] **B8.** `_resolve_content()` function (lines 626-642): **no change needed** —
      vLLM returns `content` field; fallback chain through
      `reasoning_content` → `thinking` → `reasoning` → `reason` still works
      for adaptive-thinking models.

- [ ] **B9.** `_post()` function (lines 645-650): **DELETE** — replacing with
      async vLLM backend.

- [ ] **B10.** `http_model_call()` function (lines 653-707): **REWRITE**:
      - Replace `requests.post()` → `httpx.AsyncClient` async POST.
      - Add sync wrapper using `concurrent.futures.ThreadLoopExecutor` so the
        existing `Callable[[list, dict], str] model_call` interface is preserved.
      - Change degradation loop from `while level <= 2` to `while level <= 1`.
      - Remove `reasoning_effort` from body construction.
      - Update 400 response body detection: look for `chat_template_kwargs`
        rejection only (not `reasoning_effort`).
      - Log vLLM usage tokens (`prompt_tokens`, `completion_tokens`) from
        response `usage` field.

- [ ] **B11.** FastAPI app docstring (lines 713-719): update model name and
      vLLM references.

- [ ] **B12.** `/healthz` endpoint (lines 772-820):
      - Keep `/health` probe for vLLM (same path).
      - **Remove `/props` probe** — doesn't exist on vLLM.
      - Add `/v1/models` probe to extract `vllm_model_id`.
      - Replace `context_budget` → remove `slot_tokens`, add `context_size`
        (the vLLM MAX_MODEL_LEN).
      - Add `thinking_disabled` and `context_guard` fields (matching migrated systems).

- [ ] **B13.** `__main__` block (lines 871-887): update log messages to remove
      `MODEL_PARALLEL` references and update model name.

### Phase C: Lifecycle Scripts

- [ ] **C1.** Update `start.sh`:
      - Change pre-flight model port from `8006` to `8011`.
      - Update pre-flight wait timeout to `300s` (vLLM JIT compilation needs
        up to 15 minutes on first boot).
      - Add `/v1/models` probe as vLLM health check fallback (in addition to
        `/health`).
      - Remove references to Proof-Reader's `startserver.sh` as the model
        manager; update to reference `DGXSpark_Setup/vllm-qwen/startserver.sh`.
      - Update echo messages to reflect vLLM server.

- [ ] **C2.** Update `stop.sh`:
      - Change references from "llama-server" to "vLLM server."
      - Keep behavior identical (never touches model server, only stops gateway).

### Phase D: Tests (`tests/test_extract.py`)

- [ ] **D1.** Update docstring: "v2.1" → "v3 (vLLM Migration)."

- [ ] **D2.** Add vLLM config constant tests:
      - `test_context_size_is_32768()` — assert `CONTEXT_SIZE == 32768`.
      - `test_model_name_is_vllm()` — assert `MODEL_NAME == "Qwen3.6-35B-A3B-NVFP4"`.
      - `test_model_url_is_vllm()` — assert `"8011" in MODEL_URL`.
      - `test_model_parallel_not_defined()` — assert `MODEL_PARALLEL` does not exist.
      - `test_request_timeout_120()` — assert `REQUEST_TIMEOUT == 120`.

- [ ] **D3.** Rewrite `TestBuildParams`:
      - `test_level0_no_reasoning_effort()` — assert `reasoning_effort` is
        **never** present in params (vLLM doesn't support it).
      - `test_level0_has_chat_template_kwargs()` — assert `chat_template_kwargs`
        is present at level 0.
      - `test_level1_drops_chat_template_kwargs()` — assert it's absent at level 1.
      - `test_degradation_levels_is_2()` — assert `_PARAMS_LEVELS == (0, 1)`.

- [ ] **D4.** Rewrite `TestContextGuard`:
      - `test_slot_budget_doesnt_exist()` — assert `slot_budget()` function is
        removed.
      - `test_context_budget_is_32k()` — assert `usable_prompt_room()` uses
        full 32768 budget (no division).
      - `test_check_context_error_mentions_vllm()` — assert error message
        references "vLLM context budget" not "slot budget."

- [ ] **D5.** Rewrite `TestHttpModelCall`:
      - Replace all `FakeResp` + `requests.post` monkeypatches with
        `httpx.Response` + `httpx.AsyncClient` async stubs.
      - `test_http_call_vllm_2_level_degradation()` — test 2-level (not 3-level).
      - `test_http_call_vllm_no_reasoning_effort()` — assert `reasoning_effort`
        is never in the request body.
      - `test_vllm_usage_tokens_logged()` — assert usage dict from vLLM is parsed.

- [ ] **D6.** Rewrite `TestResolveContent`:
      - No changes needed — vLLM response shape is compatible.

- [ ] **D7.** Update `TestApiSurface`:
      - `test_healthz_reports_vllm_fields()` — assert `/healthz` returns
        `vllm_model_id`, `thinking_disabled`, `context_guard` fields.
      - `test_healthz_has_no_slot_tokens_field()` — assert no `slot_tokens` in
        context budget (was present in llama.cpp version).
      - Update `test_healthz_ok_with_context_crosscheck()` to mock vLLM
        `/health` + `/v1/models` instead of llama.cpp `/health` + `/props`.

- [ ] **D8.** Update `test_defaults_sync_to_env_example()`:
      - Remove `MODEL_PARALLEL` assertion.
      - Update `MODEL_NAME` expected value to `"Qwen3.6-35B-A3B-NVFP4"`.
      - Update `LLAMA_SERVER_API_KEY` assertion to `"test_key_0000"`.
      - Update `REQUEST_TIMEOUT` expected value to `120`.
      - Add `MODEL_URL` assertion.

### Phase E: Verification

- [ ] **E1.** Run the full test suite: `pytest tests/test_extract.py -v`
      — all tests must pass.

- [ ] **E2.** Manual pre-flight check:
      - Start vLLM server (`DGXSpark_Setup/vllm-qwen/startserver.sh`).
      - Run `./start.sh` — pre-flight should confirm vLLM health on `:8011`.
      - Hit `/healthz` — verify `vllm_model_id`, `thinking_disabled`,
        `context_guard` fields present.
      - Hit `/v1/extract` with golden test case — verify 200 + valid JSON.

---

## 3. Reference: Migrated Systems

| System | Port | Model | Context | Degradation | HTTP Client |
|---|---|---|---|---|---|
| Proof-Reader | 8082 | Qwen3.6-35B-A3B-NVFP4 | 8192 | 2-level | httpx.Async |
| GDS Extraction | 8084 | Qwen3.6-35B-A3B-NVFP4 | 32768 | 2-level | httpx.Async |
| QA-Manager | 8083 | Qwen3.6-35B-A3B-NVFP4 | 32768 | 2-level | httpx.Async |
| **MDPL BOL (this)** | **8086** | **Qwen3.6-35B-A3B-NVFP4** | **32768** | **2-level** | **httpx.Async** |

All share: `test_key_0000` auth, vLLM `:8011`, `chat_template_kwargs` thinking
suppression, `httpx.AsyncClient` backend, continuous batching (no slot division).

---

## 4. Builder Notes

- **`bol_service.py` is a single file (887 lines).** All changes are in-place
  edits — no new files, no module restructure.
- **The `model_call` injection pattern** (`Callable[[list, dict], str]`) is
  intentionally preserved. Even though the production backend becomes async,
  a `ThreadPoolExecutor` wrapper ensures the callable signature stays compatible
  with `run_extract()` and `run_extract_batch()` which call `model_call(m, p)`.
- **`_resolve_content()` fallback chain** (`content` → `reasoning_content` →
  `thinking` → `reasoning` → `reason`) works with both llama.cpp and vLLM
  responses for adaptive-thinking Qwen models. No change needed.
- **`extract_json()` and `_normalize_bol()`** are pure logic — no change needed.
- **`build_prompt()` and `BOL_SYSTEM`** are domain logic — no change needed.
- **Customer table + lookup** — no change needed.
