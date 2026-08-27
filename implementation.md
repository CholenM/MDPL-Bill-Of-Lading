# Implementation Roadmap — MDPL Bill of Lading Extractor (v2.1 / "J2.1")

**Status:** Approved design — Source of Truth for the Builder.
**Pattern lineage:** Direct replication of Proof-Reader / QA-Manager / GDS-Extraction architecture.
**Spec source:** `E:\Projects\Sources\AI_MDPL_BILL_OF_LADING.md` (Japan Bill of Lading Extractor v2.1).

---

## 1. Strategic Design

### 1.1 Objectives

Build a single-file FastAPI gateway service that extracts structured shipment data from Bill of Lading (B/L) documents — supplied as **pre-extracted text** — and returns a strict JSON contract using a shared local llama.cpp model server on the DGX Spark. This is system #4 of 8; it must be operationally identical to its three siblings so ops runbooks stay uniform.

### 1.2 Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │            DGX Spark (Linux)                 │
                    │                                              │
 Caller ──HTTP──▶   │  :8086  bol_service.py   ──HTTP POST──▶  :8006 llama-server
 (x-api-key)        │  FastAPI gateway         OpenAI-compat      Qwen3.8-27B (GGUF)
                    │  - auth, validation       chat/completions   --jinja --api-key
                    │  - context guard          greedy decoding    --ctx-size 65536
                    │  - prompt fencing         think-suppress     --parallel 4
                    │  - extract_json + normalize                  (16,384 tokens/slot)
                    │  start.sh / stop.sh       (owned by Proof-Reader repo,
                    │                            NEVER started here)
                    └─────────────────────────────────────────────┘
```

OCR/PDF/image parsing is **out of scope** — callers send plain text (decision recorded below). The model is served without vision (`--mmproj`), consistent with all siblings.

### 1.3 Key Decisions (from Socratic interrogation)

| # | Question | Decision |
|---|----------|----------|
| D1 | Production port | **8086** (8082 dev PR · 8085 prod PR · 8083 QA · 8084 GDS are taken) |
| D2 | Input format | **Pre-extracted text only** (`{"bol_text": "..."}`). No file upload, no OCR dependency |
| D3 | Customer table | **Config-driven** (`.env` `CUSTOMER_TABLE`, JSON array) + **normalized tolerant matching** (case/punctuation/bracket-insensitive) |
| D4 | API surface | Mirror GDS pattern: `/v1/extract`, `/v1/extract_batch`, `/v1/version`, `/healthz`. `AssistantVersion` always `"J2.1"` |
| D5 | Context budget | One of the "remaining 5 systems = 16k each": `CONTEXT_SIZE=65536`, `MODEL_PARALLEL=4` → 16,384/slot |

### 1.4 Constraints

- **C1 — Shared model server:** Gateway must never launch/stop `llama-server`. `start.sh` polls `http://127.0.0.1:8006/health` up to 120s pre-flight and aborts if down.
- **C2 — `.env.example` == code defaults:** Enforced by a pytest test (anti-config-drift invariant, same as siblings).
- **C3 — Deterministic output:** Greedy sampling (`temperature=0`). Thinking suppression: `reasoning_effort=0` + `chat_template_kwargs{enable_thinking:false}` with 3-level degradation retry chain (cached working level).
- **C4 — Fail-closed JSON:** Unparseable model output → HTTP 503. Never return invented data.
- **C5 — Pure-function core:** All business logic takes an injectable `model_call` callable; unit-testable on Windows with a stub backend, no GPU needed.
- **C6 — Internal bearer token:** Reuse `sk-internal-proofreader` (matches shared server's `--api-key`; this is a cluster-wide constant, not a secret rotation concern).
- **C7 — Single-file service:** Entire app in `bol_service.py`, mirroring siblings. No Docker, no systemd — bash lifecycle scripts only.
- **C8 — Runtime:** Production Linux/DGX; development Windows (pytest only). Python deps identical to siblings: `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `requests>=2.31`, `python-dotenv>=1.0`; dev: `pytest>=8.0`, `httpx>=0.27`.

### 1.5 Data Contracts

**Request — `/v1/extract`**
```json
{ "bol_text": "<raw B/L document text>" }
```
Header: `x-api-key: <key>` (validated against env `API_KEYS="key:Label,..."`)

**Response — `/v1/extract` (200)**
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

**Request/Response — `/v1/extract_batch`**: identical shape to GDS-Extraction —
In: `{"entries":[{"id": "...", "bol_text": "..."}]}` · Out: `{"results":[{"id","status":"ok"|"error","data"|"error"}]}` — sequential processing, per-entry isolation (one failure never aborts the batch).

**`/v1/version`** → `{"version": "J2.1"}` (no model call).
**`/healthz`** → liveness + model-server reachability + per-slot context cross-check vs live `/props` (`"check": "ok"|"mismatch"`).

### 1.6 Error Map

| Status | Cause |
|--------|-------|
| 401 | Missing/invalid `x-api-key` |
| 422 | Malformed body · empty `bol_text` · client-side context-guard rejection (with re-provisioning guidance) |
| 503 | Model unreachable/timeout · unparseable model JSON · model-side context overflow |
| 500 | Unexpected internal error |

---

## 2. Component Design

### 2.1 Customer Lookup (`lookup_customer` — pure function)
- Load table from env `CUSTOMER_TABLE` (JSON array of `{Code, Buyer, RegisteredCustomerName}`); fall back to the spec's 4 default rows if unset.
- Normalize both sides before comparison: lowercase, strip `[ ] ( ) , .` punctuation, collapse whitespace.
- Match order: exact-normalized `RegisteredCustomerName` → exact-normalized `Buyer` → substring/containment match (document name contains registered name or vice-versa).
- **Ambiguity rule:** if multiple rows match, prefer exact-normalized hit; otherwise return `"N/A"` code but keep the extracted document `CustomerName` verbatim. Never guess between two codes.

### 2.2 Prompt Strategy (constants in `bol_service.py`)
- `BOL_SYSTEM`: role = meticulous B/L data extractor for the Japanese market; demands JSON-only output matching the exact key skeleton from §1.5; encodes field rules (B/L number alnum-only; date split into integer M/D/Y; `VoyageNumber` → `"N/A"` when undeterminable; `Brand` sourced from Marks & Numbers; cartons total integer; missing fields → `"N/A"` or null per spec §6; customer-code resolution rules referencing the embedded table).
- The active customer table is injected into the user message each request (from config, so prompt always matches live config).
- User wrapper with delimiters (injection bleed reduction):
  `Extract the following Bill of Lading. Output ONLY the required JSON object.\n\nBOL_DATA:\n<<<BOL_DATA>>>\n{bol_text}\n<<<END_BOL_DATA>>>`

### 2.3 Output Normalization (`_normalize_bol` — pure function)
Coerce parsed JSON to the §1.5 contract exactly: drop unknown keys; `BLDate` components coerced int (non-numeric → 0); `Cartons` coerced int (non-numeric → 0); string fields defaulted `"N/A"`; nested objects defaulted to their empty shape; `AssistantVersion` force-set to `"J2.1"` regardless of model output.

### 2.4 Carried-over machinery (verbatim from GDS skeleton)
`http_model_call` (degradation chain) · `_resolve_content` (content→reasoning_content→thinking→reasoning→reason) · `extract_json` (strip `<think>`/fences, first `{`..last `}`, fail-closed) · `slot_budget()`/`usable_prompt_room()`/`check_context()` (chars//3 heuristic, modes strict/warn/off) · `verify_api_key` dependency · CORS open · PID-file lifecycle.

---

## 3. Edge Cases & Failure Handling

| Edge case | Behavior |
|-----------|----------|
| B/L text exceeds ~15k slot budget | 422 pre-network rejection with actionable sizing message (strict mode) |
| Model returns `<think>` blocks / fences / preamble | Stripped by `extract_json`; resolver falls back through reasoning fields |
| Unparseable JSON after stripping | 503 fail-closed |
| Customer name not in table | `CustomerCode: "N/A"`, `CustomerName` kept verbatim |
| Ambiguous multi-row match | Prefer exact-normalized; else `"N/A"` — never guess |
| Name variants ("ANA FOODS CO., LTD." vs "Ana Foods Co., LTD") | Handled by normalizer (case/punct-insensitive) |
| Missing/undetermined voyage number | `"VoyageNumber": "N/A"` (spec §4) |
| Non-numeric cartons or date parts | Coerced to 0 by normalizer |
| Batch entry fails | That entry gets `{"status":"error","error":"..."}`; batch continues |
| Model server dies mid-run | 503; `/healthz` reports `model_server: down`; gateway stays up |
| `.env` ctx flags drift from server launch flags | `/healthz` cross-checks `/props` per-slot `n_ctx`, reports `"mismatch"` |

---

## 4. Execution Roadmap

Order matters: Types → Pure logic → Prompts → Pipeline → HTTP → Lifecycle → Tests → Docs.

- [ ] **T1** Scaffold project: create `requirements.txt`, `.gitignore`, `.env.example` (defaults: `MODEL_URL=http://127.0.0.1:8006/v1/chat/completions`, `MODEL_NAME=Qwen3.8-27B`, `LLAMA_SERVER_API_KEY=sk-internal-proofreader`, `CONTEXT_SIZE=65536`, `MODEL_PARALLEL=4`, `MODEL_MAX_TOKENS=4096`, `REQUEST_TIMEOUT=300`, `API_PORT=8086`, `API_HOST=0.0.0.0`, `API_KEYS=bol_key_0000:Default`, `CUSTOMER_TABLE=` (empty → built-in defaults), sampling params, `DISABLE_THINKING=true`, `REQUEST_TIMEOUT`, `CONTEXT_GUARD=strict`)
- [ ] **T2** Create empty `bol_service.py` shell: imports, dotenv load, env-driven config constants, Pydantic models (`ExtractRequest{bol_text}`, `BatchEntry{id?, bol_text}`, `BatchRequest{entries}`, response models per §1.5 including `BLDate`/`ShipToDestination` sub-models)
- [ ] **T3** Implement `DEFAULT_CUSTOMER_TABLE` (the 4 spec rows) + `load_customer_table()` (env JSON parse w/ safe fallback)
- [ ] **T4** Implement `_normalize_name()` + `lookup_customer(name, table)` pure function with exact → buyer → substring ordering and ambiguity guard
- [ ] **T5** Write `BOL_SYSTEM` prompt constant encoding all §4 extraction rules + JSON skeleton; implement `build_prompt(bol_text, table)` with `<<<BOL_DATA>>>` fencing and table injection
- [ ] **T6** Implement `estimate_tokens`/`slot_budget`/`usable_prompt_room`/`check_context` (chars//3 heuristic, strict/warn/off modes)
- [ ] **T7** Port `http_model_call` with 3-level thinking-suppression degradation chain (cached level), `_resolve_content` fallback chain, `extract_json` fail-closed parser — verbatim from GDS skeleton
- [ ] **T8** Implement `_normalize_bol(parsed)` → exact contract coercion (drop unknown keys, int coercion with 0 fallback, `"N/A"` defaults, force `AssistantVersion="J2.1"`)
- [ ] **T9** Implement `transform_bol(text, model_call)` orchestrator: context guard → build prompt → model call → extract_json → normalize → lookup_customer merge. Pure/injectable, no I/O
- [ ] **T10** Implement FastAPI endpoints: `POST /v1/extract`, `POST /v1/extract_batch` (sequential, per-entry isolation), `POST /v1/version`, `GET /healthz` (model `/health` ping + `/props` ctx cross-check), `GET /`, CORS, `x-api-key` dependency; map errors to §1.6 error codes
- [ ] **T11** Add `__main__` uvicorn entrypoint bound to `API_HOST:API_PORT`
- [ ] **T12** Create `start.sh` (venv bootstrap → pip install → `.env.example`→`.env` once → source → poll `:8006/health` up to 120s → launch → write `.pids` → poll own `/healthz`) and `stop.sh` (PID-file kill, scoped-name fallback, never touches model server)
- [ ] **T13** Build test suite `tests/test_extract.py` with stub backend: happy path full extraction; each field's missing/N/A fallback; date/carton coercion; customer exact/buyer/substring/no-match/ambiguous cases; custom `CUSTOMER_TABLE` loading + malformed-table fallback; context guard strict/warn/off; degradation-chain param drops; `extract_json` think-block/fence/garbage handling; batch isolation; version endpoint; `.env.example`-defaults sync pin (C2 invariant). Include golden sample under `tests/cases/` (a realistic B/L text + expected JSON)
- [ ] **T14** Run full pytest suite on Windows until green (no GPU required)
- [ ] **T15** Documentation: `README.md` (ops runbook, API docs, port map incl. 8085 prod-Proof-Reader note, context sizing section, curl examples) and `bol-prompts.md` (verbatim prompt reference) — plus update sibling READMEs' port maps if they enumerate allocations

## 5. Verification Criteria (Definition of Done)

1. `pytest tests/test_extract.py` fully green on Windows dev machine.
2. `.env.example` values byte-equal code defaults (pinned by test).
3. On DGX: `./start.sh` reaches LIVE only when `:8006` healthy; `/healthz` reports `check: ok`.
4. Golden-sample smoke test via curl returns contract-exact JSON with `AssistantVersion: "J2.1"`.
5. No code path in this repo starts or stops the shared model server.

---

## 6. Hand-off

The roadmap is ready. **Please switch to the Builder agent and say "Build this" to begin execution**, starting at T1.
