# MDPL Bill of Lading Extractor — MD-File Ingest Roadmap (v3.1)

> **Source of Truth for the Builder.** This document replans the system after the
> `GET /v1/extract → 405` incident and the OCR-team decision:
> **pipeline is `PDF → OCR (separate system) → OCR output as `.md` text file → BOL extractor`.**
> Our gateway **must receive an `.md` file** (single, small, UTF-8).
>
> Prior doc (v2.1 → v3 vLLM migration) is superseded for API scope but remains
> valid for backend (vLLM `:8011`, `Qwen3.6-35B-A3B-NVFP4`, 2-level degradation,
> continuous batching). Do not regress it.

---

## 1. Strategic Design

### 1.1 Objectives

1. **Explain and close the 405:** `GET /v1/extract` is *correct* `405 Method Not Allowed`
   — route is `POST`-only by design (`bol_service.py:854`). No GET handler will be added.
   Fix is client-side + docs.
2. **Support hardwired OCR handoff:** add `POST /v1/extract_file` accepting
   `multipart/form-data` with a single `.md` (allow `.txt`/`.markdown` alias) file,
   UTF-8, small. Reuse the exact same pure pipeline (`run_extract` →
   `extract_json` → `_normalize_bol` → `lookup_customer`) so J2.1 output is
   bit-identical to `POST /v1/extract`.
3. **Preserve backward compatibility:** keep `POST /v1/extract` (`{bol_text}`),
   `POST /v1/extract_batch`, `POST /v1/version`, `GET /healthz`, `GET /` unchanged
   in behavior. Only `GET /` endpoint list grows by one entry.
4. **Stay gateway-only:** never start/stop the shared vLLM server (`:8011`).
   No model, port, or schema change.

### 1.2 Architecture (Before → After)

**Before (v3 JSON-only — causes 405 + 422 for file-uploaders):**

```text
PDF → [external OCR system] → ocr_output.md (on OCR host)
                                          |
                     GET /v1/extract  →  405 (no GET route)
              POST text/plain --data-binary @ocr.md → 422 (no JSON body)
              POST -F file=@ocr.md → 422 (no bol_text field)
              POST JSON {"bol_text": "<md contents>"} → 200 (only working path)
```

**After (v3.1 — both paths work, same core):**

```mermaid
flowchart LR
    PDF[PDF Bill of Lading] --> OCR[External OCR System<br/>separate, owned by colleague]
    OCR --> MD[ocr_output.md<br/>single small UTF-8]
    MD -->|Option A: JSON wrapper<br/>client reads file to string| JSON[POST /v1/extract<br/>application/json {bol_text}]
    MD -->|Option B: direct file<br/>hardwired uploader| FILE[POST /v1/extract_file<br/>multipart/form-data file=.md]
    JSON --> GW[FastAPI Gateway :8086<br/>bol_service.py]
    FILE --> GW
    GW -->|run_extract same path| VLLM[vLLM :8011<br/>Qwen3.6-35B-A3B-NVFP4]
    VLLM --> OUT[J2.1 shipment JSON]
```

**Request flow inside gateway (both endpoints converge):**

```text
POST /v1/extract         ─┐
  {bol_text: str}         ├─→ load_customer_table() → run_extract({bol_text}, table, http_model_call)
POST /v1/extract_file    ─┘         ↑                                              ↑
  file: UploadFile (.md) ── _read_md_upload() ── {bol_text: decoded str} ─────────┘
                                                    (build_prompt → check_context → model_call → extract_json)
```

### 1.3 API Definitions

All endpoints require `x-api-key` header except `GET /` and `GET /healthz`.
Interactive docs at `/docs`. `GET /v1/extract*` intentionally does **not** exist.

| Endpoint | Method | Request | Success | Notes |
|---|---|---|---|---|
| `/` | GET | — | `200 {service, version, endpoints, docs}` | Add `/v1/extract_file` to `endpoints` list |
| `/healthz` | GET | — | `200 {status, model_server, ..., version}` | Unchanged |
| `/v1/extract` | POST | `application/json {"bol_text": string(min 1)}` | `200 J2.1 JSON` | Unchanged; `GET` → `405` (expected) |
| `/v1/extract_file` | **POST (NEW)** | `multipart/form-data` single field `file: (.md/.txt/.markdown)` UTF-8, non-empty, size-capped | `200 J2.1 JSON` (identical schema) | New; `GET` → `405` (expected) |
| `/v1/extract_batch` | POST | `{"entries": [{id?, bol_text}]}` | `200 {results: [{id, status, data/error}]}` | Unchanged |
| `/v1/version` | POST | — | `200 {"version": "J2.1"}` | Unchanged; `GET` → `405` (expected) |

**NEW `POST /v1/extract_file` contract:**

```bash
# happy path
curl -s http://127.0.0.1:8086/v1/extract_file \
  -H "x-api-key: $KEY" \
  -F "file=@ocr_output.md;type=text/markdown"
# → 200 + J2.1 JSON (same shape as /v1/extract)

# PowerShell (OCR host)
Invoke-RestMethod -Uri "$BASE/v1/extract_file" -Method Post `
  -Headers @{"x-api-key"=$KEY} -Form @{file=Get-Item ./ocr_output.md}
```

Server behavior:

1. Require field name exactly `file` (FastAPI `File(...)` → missing → `422`).
2. Validate extension allowlist: `.md`, `.markdown`, `.txt` (case-insensitive).
   Reject others → `422 detail: "Unsupported file type ..."`.
3. Size guard: `MAX_UPLOAD_CHARS = usable_prompt_room() * 3` chars
   (≈ `28216*3 = 84648` chars at defaults). Reject larger → `422` with same
   guidance style as `ContextGuardExceeded`. No new `.env` var (derived, avoids
   config-drift test churn).
4. Decode `await file.read()` as UTF-8 strict. On `UnicodeDecodeError` →
   `422 detail: "File must be UTF-8 ..."`. No `shift_jis` fallback (per
   interrogation: single small UTF-8 only; add fallback only if later proven needed).
5. Strip; reject empty/whitespace-only → `422`.
6. Call `run_extract({"bol_text": text}, table, http_model_call)` — identical
   downstream: `check_context` (`422`), `ModelUnavailable`/`ContextExceeded` (`503`),
   unexpected (`500`). Log `filename + char count` (never file contents at INFO).

### 1.4 Data Schemas

**J2.1 output (unchanged, `_normalize_bol` is authoritative):**

```json
{
  "AssistantVersion": "J2.1",
  "BLNumber": "string",
  "BLDate": { "Month": 0, "Day": 0, "Year": 0 },
  "CustomerName": "string",
  "CustomerCode": "string",
  "CustomerAddress": "string",
  "Shipper": "string",
  "ShipperAddress": "string",
  "ShipToDestination": { "City": "string", "Country": "string" },
  "ShipVia": "string",
  "VoyageNumber": "string",
  "Brand": "string",
  "PortOfOrigin": "string",
  "Cartons": 0
}
```

Placeholders per spec §6: `"N/A"` strings, `0` numerics, `AssistantVersion` always
`"J2.1"`. Gateway `lookup_customer` overrides model `CustomerCode`.

**Inputs:**

```python
# existing — unchanged
class ExtractRequest(BaseModel):
    bol_text: str = Field(..., min_length=1)

# new — NOT a pydantic model; FastAPI UploadFile signature:
# async def extract_file(file: UploadFile = File(...), _api_key: str = Depends(verify_api_key))
# validation lives in helper _read_md_upload(file) -> str
```

`.md` contents are opaque text (markdown `#`, `|`, `**` harmless inside
`<<<BOL_DATA>>>...<<<END_BOL_DATA>>>` fencing in `build_prompt`).

### 1.5 Constraints (from Interrogation + Debugger RCA)

- **C1. POST-only is intentional.** `GET /v1/extract`, `GET /v1/extract_file`,
  `GET /v1/version` must all remain `405`. Reason: GET cannot carry `bol_text`
  reliably (no body semantics, URL length 2–8k, PII in logs/cache). Fix is client
  uses POST; docs must state this explicitly (closes 405 confusion).
- **C2. Input is single small UTF-8 `.md`.** No PDF/image bytes to this gateway
  (OCR is upstream, separate system). No batch-file endpoint in v3.1; batch
  callers use existing `POST /v1/extract_batch` with strings.
- **C3. Reuse, don't fork.** `POST /v1/extract_file` must call `run_extract`
  (same `build_prompt`/`check_context`/`extract_json`/`lookup_customer`). No
  duplicate prompt logic, no schema fork.
- **C4. Gateway-only, shared vLLM.** No change to `MODEL_URL (:8011)`,
  `MODEL_NAME (Qwen3.6-35B-A3B-NVFP4)`, `CONTEXT_SIZE (32768)`, `API_PORT (8086)`,
  `REQUEST_TIMEOUT (120)`, 2-level `chat_template_kwargs` degradation.
- **C5. Backward compatible.** Existing clients/tests for `/v1/extract` keep
  passing untouched. `test_defaults_sync_to_env_example` must keep passing
  (hence no new required `.env` var).
- **C6. Auth + status ladder preserved:** `GET wrong method → 405` (before auth);
  `POST no/bad key → 401`; `POST bad body/file → 422`; `POST good but guard/model
  fail → 422/503`; unexpected → `500`. New endpoint mirrors this exactly.
- **C7. Dependency minimal:** only add `python-multipart` (required by Starlette
  `UploadFile` form parsing). `httpx`, `requests`, `fastapi`, `uvicorn` unchanged.
- **C8. Windows-dev testable:** full suite passes on Windows with stub
  `http_model_call` (no GPU/vLLM), same as today (`pytest -v`).

### 1.6 Edge Cases & Failure Modes

| Scenario | Handling |
|---|---|
| `GET /v1/extract` or `GET /v1/extract_file` | `405 Method Not Allowed` + `Allow: POST` (Starlette default). Documented as expected; triage: use POST. |
| Missing `file` field / wrong field name | `422` (FastAPI validation). |
| Wrong extension (`.pdf`, `.png`, `.docx`, no ext) | `422 "Unsupported file type ... use .md/.txt"`. Never sniff/convert PDF. |
| Empty file (0 bytes) or whitespace-only `.md` | `422` (mirrors `min_length=1` + `run_extract` whitespace check). |
| Non-UTF-8 bytes | `422 "File must be UTF-8 ..."`. No silent mojibake. |
| Oversized `.md` (> `usable_prompt_room()*3` chars) | `422` before any model call, with sizing guidance (mirrors `ContextGuardExceeded`; strict/warn/off honors `CONTEXT_GUARD`). |
| Prompt still over budget after file passes size cap (system prompt + table overhead) | `422` from existing `check_context` (strict) — defense in depth. |
| vLLM down / timeout / unparseable JSON | `503` via existing `ModelUnavailable`/`ContextExceeded` mapping. |
| Missing/invalid `x-api-key` on new endpoint | `401` via existing `verify_api_key`. |
| Markdown syntax / Japanese / newlines in `.md` | Harmless; passed verbatim inside `<<<BOL_DATA>>>` fences. |
| Filename with path traversal / weird chars | Ignored except for logging + extension check; never written to disk. |
| Large `filename` log injection | Log filename only, sanitized to basename, at INFO; never log file contents. |

---

## 2. The Execution Roadmap (Task List)

> Sequential. Each item atomic + verifiable. Do not skip verification phases.
> All paths absolute to repo root `E:\Projects\MDPL-Bill-Of-Lading`.

### Phase A: Dependencies & Config

- [ ] **A1.** Add `python-multipart>=0.0.9` to `requirements.txt` runtime deps
      (after `httpx` line). Verify: `pip install -r requirements.txt` succeeds on
      Windows; `python -c "import multipart"` passes.
- [ ] **A2.** Verify no `.env.example` change needed (no new var; upload cap is
      derived). Verify: `test_defaults_sync_to_env_example` still passes unmodified.

### Phase B: Core Service (`bol_service.py` — in-place edits only)

- [ ] **B1.** Imports: add `UploadFile`, `File` to the existing
      `from fastapi import ...` import (line 45). Verify: `python -c "import bol_service"` passes.
- [ ] **B2.** Implement helper `_read_md_upload(file: UploadFile) -> str` (place
      just above `# --- Endpoints ---`, ~line 799):
      - `ALLOWED_EXTS = {".md", ".markdown", ".txt"}` (lowercased suffix check via `os.path.splitext`).
      - Read `await file.read()`; enforce `len(bytes) <= usable_prompt_room()*3` (bytes ≈ chars pre-decode; cheap pre-check) else raise `ValueError` with sizing guidance.
      - Decode UTF-8 strict; `UnicodeDecodeError` → `ValueError("File must be UTF-8-encoded markdown ...")`.
      - Enforce decoded `len(text) <= usable_prompt_room()*3` else `ValueError`.
      - Reject empty/whitespace-only → `ValueError("'bol_text' must be a non-empty string." style message)`.
      - Return decoded `str`. Pure async helper, no model call.
      - Verify: unit-importable; manual stub test with fake `UploadFile`.
- [ ] **B3.** Implement `POST /v1/extract_file` endpoint (place immediately after
      existing `extract()` at ~line 872, before `extract_batch`):
      ```python
      @app.post("/v1/extract_file", response_class=JSONResponse, tags=["Extract"])
      async def extract_file(_api_key: str = Depends(verify_api_key), file: UploadFile = File(...)):
          table = load_customer_table()
          try:
              text = await _read_md_upload(file)
              record = run_extract({"bol_text": text}, table, http_model_call)
          except ValueError as exc:
              raise HTTPException(status_code=422, detail=str(exc))
          except (ModelUnavailable, ContextExceeded) as exc:
              raise HTTPException(status_code=503, detail=str(exc))
          except Exception:
              logger.error("Unexpected extract_file error", exc_info=True)
              raise HTTPException(status_code=500, detail="Processing failed.")
          return JSONResponse(content=record)
      ```
      - Log `filename + chars` on success (basename only).
      - Verify: `/docs` shows the endpoint; `GET /v1/extract_file` returns `405`.
- [ ] **B4.** Update `root()` (`@app.get("/")` ~line 800) `endpoints` list to include
      `"/v1/extract_file"`. Verify: `TestClient.get("/").json()["endpoints"]` contains it.
- [ ] **B5.** Update module docstring (lines 1–34) + FastAPI `description`
      (~line 753): note both `POST /v1/extract (JSON)` and
      `POST /v1/extract_file (.md upload)` converge on same pipeline; restate
      `GET → 405 by design`. Verify: visual diff only, no logic change.

### Phase C: Tests (`tests/test_extract.py` — append, never weaken existing)

- [ ] **C1.** Add `TestExtractFile` class (stub backend via existing `client` fixture):
      - `test_file_happy_path_parity`: POST golden `.txt` bytes as
        `files={"file": ("ocr.md", open(cases/bol_golden_input.txt,"rb"), "text/markdown")}`
        with auth → `200` + body == `_golden()` (parity with `/v1/extract`).
      - `test_file_missing_401`: no auth → `401`.
      - `test_file_missing_field_422`: `data={}` no file → `422`.
      - `test_file_wrong_extension_422`: `("scan.pdf", b"%PDF", "application/pdf")` → `422`.
      - `test_file_empty_422`: `("empty.md", b"   ", "text/markdown")` → `422`.
      - `test_file_non_utf8_422`: `("bad.md", b"\xff\xfe\x00", ...)` → `422`.
      - `test_file_oversize_422`: monkeypatch tiny budget or post huge payload → `422`.
      - `test_file_503_on_bad_model`: monkeypatch `http_model_call → "no JSON"` → `503`.
      - `test_get_on_file_405`: `client.get("/v1/extract_file")` → `405`.
      - `test_get_on_extract_405_regression`: `client.get("/v1/extract")` → `405` (locks the RCA).
      - Verify: `pytest tests/test_extract.py -v` all green (existing + ~10 new).
- [ ] **C2.** Update `test_root_lists_endpoints` (line 685–691) to also assert
      `"/v1/extract_file"` in payload. Verify: passes.

### Phase D: Docs (keep code + docs in sync)

- [ ] **D1.** `README.md`: add `POST /v1/extract_file` section with `curl` + PowerShell
      examples; add `GET → 405 is expected, use POST` troubleshooting row in Errors
      table; add one-line pipeline note `PDF → OCR → .md → POST /v1/extract_file`.
      Verify: commands copy-paste runnable.
- [ ] **D2.** `bol-prompts.md`: append one paragraph — file path is byte-identical
      prompt path (decoded text goes into same `<<<BOL_DATA>>>` fence); no prompt
      change. Verify: no other edits.
- [ ] **D3.** (Optional, only if drift found) sync `.env.example` comments to mention
      `.md` cap derived from `CONTEXT_SIZE`. No new keys. Verify: config-drift test passes.

### Phase E: Verification (DGX + Windows)

- [ ] **E1.** Windows: `pytest -v` — entire suite green.
- [ ] **E2.** Windows smoke (stub): `TestClient` POST golden file → `200`;
      `GET /v1/extract` → `405`; `POST -F file=@...txt` raw without JSON still `422` on old endpoint (proves new endpoint was the missing piece).
- [ ] **E3.** DGX pre-flight: `./start.sh` → vLLM `:8011` healthy → gateway `:8086`
      `/healthz` healthy → `/docs` lists both extract endpoints.
- [ ] **E4.** DGX golden file test:
      ```bash
      KEY=bol_key_0000
      curl -s http://127.0.0.1:8086/v1/extract_file -H "x-api-key: $KEY" \
        -F "file=@tests/cases/bol_golden_input.txt;type=text/markdown" | python3 -m json.tool
      # expect 200 + AssistantVersion J2.1 + BLNumber YKO2604155
      ```
- [ ] **E5.** DGX regression: `curl -i http://127.0.0.1:8086/v1/extract` → `405 + Allow: POST`;
      `curl -s POST /v1/extract -d '{"bol_text": "..."}'` still `200` (backward compat).
- [ ] **E6.** OCR handoff checklist to colleague: give her `E4` curl + PowerShell
      `-Form @{file=...}` snippet, `x-api-key`, `:8086` base URL, and
      `healthz → 200` pre-check. Confirm her uploader uses `POST multipart field=file`.

---

## 3. Reference

| Item | Value |
|---|---|
| Gateway port | `8086` (`API_PORT`, `bol_service.py:86`) |
| vLLM | `:8011`, `Qwen3.6-35B-A3B-NVFP4`, `CONTEXT_SIZE=32768` |
| Prompt room | `32768-4096-256 = 28216 tokens ≈ 84648 chars cap` |
| Auth | `x-api-key: bol_key_0000` (dev) |
| Golden | `tests/cases/bol_golden_input.txt` → `bol_golden_expected.json` (BL `YKO2604155`) |
| Siblings | Proof-Reader `:8082/:8085`, QA `:8083`, GDS `:8084` — all JSON-in; this is first to add file ingest |

## 4. Builder Notes

- **Single-file app (918 lines).** All backend changes in-place in `bol_service.py`;
  no new modules, no restructure.
- **`model_call` injection preserved** (`Callable[[list, dict], str]`); new endpoint
  is `async` only for `await file.read()`, then calls sync `run_extract` unchanged.
- **`python-multipart` is mandatory** — without it every `POST /v1/extract_file`
  returns `500`/`422` form-parse error. `start.sh` installs from `requirements.txt`
  automatically.
- **Never write uploads to disk.** Keep in memory; enforce size cap pre- and
  post-decode to avoid OOM on malicious large files.
- **Do not add GET handlers, do not change J2.1 schema, do not touch vLLM server.**
- **Order:** A → B → C → D → E. Stop and report if any `pytest` fails.
