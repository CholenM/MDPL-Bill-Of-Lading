# AI MDPL Bill of Lading Extractor (v2.1 / "J2.1")

Lean, JSON-in / JSON-out **Bill of Lading Extractor** for the Japan market.
Parses pre-extracted B/L document text into the standardized shipment JSON and
runs fully offline on the NVIDIA DGX Spark by reusing the already-running
llama.cpp model server. This service is a **gateway only** — it never starts or
stops the model server.

Architecture mirrors the sibling AI systems: Proof-Reader (:8082 dev / :8085
prod), QA-Manager (:8083), GDS-Extraction (:8084).

> **Input scope:** pre-extracted plain text only. PDF/image/OCR parsing is
> deliberately upstream (the OCR pipeline project). Qwen3.8-27B runs without
> vision (`--mmproj`), same as all siblings.

---

## Quick Start (DGX Spark)

```bash
./start.sh        # bootstraps venv/.env once, pre-flights model server, launches gateway
./stop.sh         # stops ONLY this gateway — never touches llama-server
```

`start.sh` requires the shared model server to be healthy first. If it isn't,
start it from the Proof-Reader project (`./startserver.sh`) and retry.

## Port Map

| Port | Service |
|------|---------|
| 8001 / 8080 | OCR pipeline (other system) |
| 8006 | Shared llama-server (Qwen3.8-27B) — owned by Proof-Reader's `startserver.sh` |
| 8082 | Proof-Reader gateway (dev) |
| 8083 | QA-Manager gateway |
| 8084 | GDS-Extraction gateway |
| **8085** | **Proof-Reader gateway (PROD)** — moved from 8082 due to a port conflict |
| **8086** | **This gateway** (Bill of Lading Extractor) |

---

## API

All endpoints require the `x-api-key` header (keys configured via `API_KEYS`
in `.env`). Interactive docs at `/docs`.

### `POST /v1/extract`

```bash
curl -s http://127.0.0.1:8086/v1/extract \
  -H "x-api-key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"bol_text": "BILL OF LADING\nB/L NO. YKO2604155\n..."}'
```

Response (200), contract-exact:

```json
{
  "AssistantVersion": "J2.1",
  "BLNumber": "YKO2604155",
  "BLDate": { "Month": 4, "Day": 15, "Year": 2026 },
  "CustomerName": "ANA FOODS CO., LTD",
  "CustomerCode": "C0003",
  "CustomerAddress": "...",
  "Shipper": "...",
  "ShipperAddress": "...",
  "ShipToDestination": { "City": "Tokyo", "Country": "Japan" },
  "ShipVia": "MSC ARIES",
  "VoyageNumber": "052E",
  "Brand": "ANAFU",
  "PortOfOrigin": "YOKOHAMA, JAPAN",
  "Cartons": 1200
}
```

Placeholders per spec §6: `"N/A"` for undetermined strings, `0` for undetermined
numerics. `AssistantVersion` is ALWAYS `"J2.1"` regardless of document content.

### `POST /v1/extract_batch`

```json
{ "entries": [ { "id": "bl-001", "bol_text": "..." }, { "bol_text": "..." } ] }
```

Sequential processing with per-entry isolation; overall status stays 200:

```json
{ "results": [ { "id": "bl-001", "status": "ok", "data": { ... } },
               { "id": null,   "status": "error", "error": "..." } ] }
```

### Other endpoints

| Endpoint | Method | Notes |
|----------|--------|-------|
| `/v1/version` | POST | `{"version": "J2.1"}` — never touches the model |
| `/healthz` | GET | Liveness + model reachability + context-budget cross-check vs server `/props` |

### Errors

| Status | Cause |
|--------|-------|
| 401 | Missing/invalid `x-api-key` |
| 422 | Malformed body · empty `bol_text` · prompt exceeds slot budget (with re-provisioning guidance) |
| 503 | Model unreachable/timeout · unparseable model JSON (fail-closed) |
| 500 | Unexpected internal error |

---

## Registered Customer Lookup

The customer code table is **config-driven** (decision D3). Default rows come
from the v2.1 spec (C0002/C0003/C0005/C0006). To add or change customers
without touching code, set `CUSTOMER_TABLE` in `.env` as a JSON array:

```json
CUSTOMER_TABLE="[{"Code":"C0009","Buyer":"New Co.","RegisteredCustomerName":"New Co., Ltd."}]"
```

Matching is tolerant: case-, punctuation-, and bracket-insensitive
(`"ANA FOODS CO., LTD."` == `"Ana Foods Co., LTD"`). Match order:
exact normalized RegisteredCustomerName → exact normalized Buyer → substring
containment. Ambiguity guard: if two different codes match, the result is
`"N/A"` — never a guess. The gateway-side lookup is authoritative (it overrides
whatever code the model produced).

---

## Context Sizing

Per the allocation plan: Proof-Reader 8k · GDS-Extraction 16k · QA-Manager
100k · remaining five systems 16k each. This system runs the **16k profile**:

```
CONTEXT_SIZE=65536   MODEL_PARALLEL=4   → 16,384 tokens/slot
```

llama.cpp divides its total `--ctx-size` across `--parallel` slots, so these
values MUST mirror Proof-Reader's `startserver.sh` launch flags. `/healthz`
cross-checks against the live server's per-slot `n_ctx` and reports
`check: mismatch` on drift. The client-side guard (chars//3 heuristic, modes
strict/warn/off) rejects over-budget prompts with HTTP 422 before any network
call.

Prompt room = slot budget − max_tokens − 256 safety margin ≈ 12,032 tokens —
comfortably above typical B/L text.

---

## Configuration

Copy `.env.example` → `.env` (done automatically by `start.sh`). Key settings:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_URL` | `http://127.0.0.1:8006/v1/chat/completions` | Shared llama-server endpoint |
| `MODEL_NAME` | `Qwen3.8-27B` | Must match the loaded model |
| `LLAMA_SERVER_API_KEY` | `sk-internal-proofreader` | Bearer token; must match server `--api-key` |
| `CONTEXT_SIZE` / `MODEL_PARALLEL` | `65536` / `4` | Slot budget math; mirror server flags |
| `MODEL_MAX_TOKENS` | `4096` | Output cap |
| `MODEL_TEMP` | `0.0` | Greedy decoding for determinism |
| `DISABLE_THINKING` | `1` | Thinking suppression + degradation chain |
| `REQUEST_TIMEOUT` | `300` | Seconds per model call |
| `CONTEXT_GUARD` | `strict` | `strict` / `warn` / `off` |
| `CUSTOMER_TABLE` | *(empty → spec defaults)* | Config-driven customer rows |
| `API_PORT` | `8086` | Gateway port |
| `API_KEYS` | `bol_key_0000:BOL Extraction Local Testing` | Client keys (`key:Label,...`) |

A test pins `.env.example` values == code defaults (anti-config-drift).

---

## Development & Testing

Everything is developed/tested on Windows with a stub backend — no GPU needed:

```bash
pytest -v            # 82 tests, all green
```

Golden sample lives in `tests/cases/` (realistic Japan-market B/L + expected
JSON). All business logic is pure functions taking an injectable
`model_call`, so the full pipeline is exercised without llama.cpp.

On-DGX smoke test after deploy:

```bash
curl -s http://127.0.0.1:8086/healthz
# expect: {"status":"healthy","context_budget":{"check":"match",...},...}

curl -s http://127.0.0.1:8086/v1/extract \
  -H "x-api-key: <KEY>" -H "Content-Type: application/json" \
  -d "{\"bol_text\": $(python3 -c 'import json;print(json.dumps(open("tests/cases/bol_golden_input.txt").read()))')}"
```

## Project Layout

```
MDPL-Bill-Of-Lading/
├── bol_service.py            # entire app (FastAPI + pipeline + LLM client)
├── .env.example              # config template (== code defaults)
├── requirements.txt
├── start.sh / stop.sh        # gateway-only lifecycle
├── implementation.md         # design decisions + roadmap (source of truth)
├── README.md                 # this file
├── bol-prompts.md            # verbatim prompt reference
└── tests/
    ├── test_extract.py       # pytest suite (stub backend)
    └── cases/
        ├── bol_golden_input.txt
        └── bol_golden_expected.json
```

## Version History

| Version | Date | Description |
|---------|------|-------------|
| J2.1 | 2026-08 | Local-model migration (DGX Spark / Qwen3.8-27B). Gateway port 8086; config-driven customer table with tolerant matching; batch endpoint added. Based on spec v2.1 (originally gpt-4o/gpt-4o-mini). |
