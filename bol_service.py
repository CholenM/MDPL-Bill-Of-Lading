"""
AI MDPL Bill of Lading Extractor — FastAPI Service (v2.1 / "J2.1")
======================================================================
Lean, JSON-in / JSON-out **Bill of Lading Extractor** for the Japan market.
Parses pre-extracted B/L document text into the standardized shipment JSON
defined in the project spec (AssistantVersion J2.1), and runs fully offline on
the NVIDIA DGX Spark by reusing the already-running llama.cpp (CUDA) model
server. This service is a **gateway only** — it never starts or stops the model
server (see start.sh / stop.sh).

Document parsing (PDF/image/OCR) is deliberately OUT OF SCOPE: callers send
pre-extracted plain text, matching the Proof-Reader / QA-Manager /
GDS-Extraction siblings (Qwen3.8 is served without vision).

Architecture (mirrors the sibling skeleton):

    Client  --POST /v1/extract-->  FastAPI gateway (:8086)
            {bol_text}
            {entries: [{id, bol_text}]}
                                           |
                                           v
                                      llama-server (:8006, shared server)
                                      Qwen3.8-27B, CUDA, text mode (--jinja)

The core logic (build_prompt / build_params / estimate_tokens / check_context /
extract_json / lookup_customer / _normalize_bol / run_extract /
run_extract_batch) are pure functions taking an injectable ``model_call``
callable, so the entire business logic is unit-tested on the Windows dev
machine with a stub backend — no GPU needed.

Usage:
    python bol_service.py                  # reads config from .env
    uvicorn bol_service:app                # alternative via uvicorn CLI
"""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Callable, Optional, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Header, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import requests

# ---------------------------------------------------------------------------
# Configuration — all values sourced from .env (see .env.example).
#
# NOTE: CONTEXT_SIZE and MODEL_PARALLEL mirror the SHARED server's launch flags
# (Proof-Reader's startserver.sh). They exist here only to compute the local
# per-slot context budget for the client-side guard and the /healthz cross-check.
# ---------------------------------------------------------------------------
load_dotenv()

CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "65536"))
MODEL_PARALLEL = int(os.getenv("MODEL_PARALLEL", "4"))
MODEL_URL = os.getenv("MODEL_URL", "http://127.0.0.1:8006/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3.8-27B")
# Internal bearer token the gateway presents to the shared llama-server. Sent
# ONLY if non-empty. THIS MUST MATCH the shared server's --api-key, or the
# server rejects the call with a 401 (the gateway then returns a 503 on every
# extract). The shared server is started with --api-key sk-internal-proofreader.
LLAMA_SERVER_API_KEY = os.getenv("LLAMA_SERVER_API_KEY", "sk-internal-proofreader")
# Greedy decoding (temp=0) for run-to-run determinism on factual document
# extraction.
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "0.0"))
MODEL_TOP_P = float(os.getenv("MODEL_TOP_P", "0.5"))
MODEL_TOP_K = int(os.getenv("MODEL_TOP_K", "40"))
MODEL_MAX_TOKENS = max(64, int(os.getenv("MODEL_MAX_TOKENS", "4096")))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "1").lower() == "1"
# Client-side context guard:
#   strict (default) -> reject over-budget requests with a 422 + guidance
#   warn             -> allow the request but log a warning
#   off              -> skip the guard entirely (for testing only)
CONTEXT_GUARD = os.getenv("CONTEXT_GUARD", "strict").strip().lower()
API_KEY_AUTH_HEADER = os.getenv("API_KEY_AUTH_HEADER", "x-api-key").lower()
# Deliberately 8086 — 8082 Proof-Reader dev, 8085 Proof-Reader PROD (moved off
# 8082 due to a port conflict), 8083 QA-Manager, 8084 GDS-Extraction.
API_PORT = int(os.getenv("API_PORT", "8086"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")

# Config-driven registered-customer table (JSON array). Empty => built-in
# defaults from the v2.1 spec.
CUSTOMER_TABLE_ENV = os.getenv("CUSTOMER_TABLE", "").strip()

# The documented assistant/service version (spec v2.1).
VERSION = "J2.1"

# Tokens reserved beyond the output budget as headroom (safety margin).
_SAFETY_MARGIN = 256
# Degradation levels for chat-template / reasoning suppression.
# Level 0 = full params, level 1 drops chat_template_kwargs,
# level 2 drops reasoning_effort on top of level 1.
_PARAMS_LEVELS = (0, 1, 2)
# Last known-working degradation level, cached to avoid retrying on every call.
_PARAMS_LEVEL = 0

# Sampling parameters forwarded to the model request. Recognized keys must also
# be accepted by llama.cpp's OpenAI-compatible API.
_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "max_tokens",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bol_service")

# ---------------------------------------------------------------------------
# Lean API-key database — loaded from .env, kept in memory.
# Format: key1:Label1,key2:Label2,...  (labels are cosmetic / for logging)
# ---------------------------------------------------------------------------
DEFAULT_KEYS = "bol_key_0000:BOL Extraction Local Testing"


def _parse_api_keys(raw: str) -> dict[str, str]:
    keys: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", 1)
        key = parts[0].strip()
        if key:
            keys[key] = parts[1].strip() if len(parts) > 1 else ""
    return keys


API_KEY_DB = _parse_api_keys(os.getenv("API_KEYS", DEFAULT_KEYS))
logger.info(f"Loaded {len(API_KEY_DB)} API key(s): {list(API_KEY_DB.values()) or '(none)'}")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ModelUnavailable(RuntimeError):
    """Raised when the model server cannot satisfy a request."""


class ContextExceeded(RuntimeError):
    """Raised when the model server rejects the request because it overflows
    the (fixed) context window. Maps to HTTP 503 (server-side)."""


class ContextGuardExceeded(ValueError):
    """Raised by the client-side guard before any network call when the prompt
    cannot fit in a slot. Maps to HTTP 422 (client-side, actionable)."""


# ===========================================================================
# REGISTERED CUSTOMER TABLE  (config-driven; D3 decision)
# ===========================================================================
# Built-in defaults from the v2.1 spec §3. Used when CUSTOMER_TABLE env is
# unset/empty or unparseable. Ops can extend/replace rows via .env without a
# code change.
DEFAULT_CUSTOMER_TABLE: list[dict] = [
    {"Code": "C0002", "Buyer": "Laysun [Far East] Limited", "RegisteredCustomerName": "Laysun [Far East] Limited"},
    {"Code": "C0003", "Buyer": "ANA Foods Co., LTD", "RegisteredCustomerName": "ANA Foods Co., LTD"},
    {"Code": "C0005", "Buyer": "Hiro International", "RegisteredCustomerName": "Hiro International"},
    {"Code": "C0006", "Buyer": "Farmind Corporation", "RegisteredCustomerName": "Farmind Corporation"},
]


def load_customer_table(raw: Optional[str] = None) -> list[dict]:
    """Parse the CUSTOMER_TABLE env JSON into validated rows.

    Falls back to DEFAULT_CUSTOMER_TABLE on empty/unparseable/malformed input
    (never raises — a bad config must not take the service down).
    """
    source = CUSTOMER_TABLE_ENV if raw is None else raw
    if not source:
        return [dict(r) for r in DEFAULT_CUSTOMER_TABLE]
    try:
        data = json.loads(source)
    except json.JSONDecodeError as exc:
        logger.warning(f"CUSTOMER_TABLE is not valid JSON ({exc}); using defaults.")
        return [dict(r) for r in DEFAULT_CUSTOMER_TABLE]
    if not isinstance(data, list):
        logger.warning("CUSTOMER_TABLE is not a JSON array; using defaults.")
        return [dict(r) for r in DEFAULT_CUSTOMER_TABLE]
    rows: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        code = str(item.get("Code", "")).strip()
        buyer = str(item.get("Buyer", "")).strip()
        reg = str(item.get("RegisteredCustomerName", "")).strip()
        if code and (buyer or reg):
            rows.append({"Code": code, "Buyer": buyer, "RegisteredCustomerName": reg})
    if not rows:
        logger.warning("CUSTOMER_TABLE contained no valid rows; using defaults.")
        return [dict(r) for r in DEFAULT_CUSTOMER_TABLE]
    return rows


def _normalize_name(name: Any) -> str:
    """Case/punctuation/bracket-insensitive normalization for tolerant matching.

    Lowercases, strips [ ] ( ) , . & ' punctuation and collapses whitespace so
    e.g. "ANA FOODS CO., LTD." matches "Ana Foods Co., LTD".
    """
    s = str(name or "").lower()
    s = re.sub(r"[\[\]\(\)\{\}\.,&'\"`~\-_/\\]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def lookup_customer(name: str, table: list[dict]) -> str:
    """Resolve a document CustomerName to its registered customer Code.

    Match order (D3 decision):
      1. exact normalized RegisteredCustomerName
      2. exact normalized Buyer
      3. substring containment either direction (document contains registered
         name or vice-versa)

    Ambiguity guard: never guess between two different codes. If step 1 or 2
    yields exactly one row, that row wins. Otherwise, collect containment
    candidates; if they all share ONE distinct code, use it; anything ambiguous
    or unmatched returns "N/A" (the caller keeps the verbatim CustomerName).
    """
    target = _normalize_name(name)
    if not target:
        return "N/A"

    # 1. Exact normalized RegisteredCustomerName
    hits = [r for r in table if _normalize_name(r.get("RegisteredCustomerName")) == target]
    # 2. Exact normalized Buyer
    if not hits:
        hits = [r for r in table if _normalize_name(r.get("Buyer")) == target]

    codes = {r.get("Code") for r in hits}
    if len(codes) == 1:
        return next(iter(codes))

    # 3. Substring containment either direction
    candidates: dict[Any, None] = {}
    for r in table:
        reg_n = _normalize_name(r.get("RegisteredCustomerName"))
        buy_n = _normalize_name(r.get("Buyer"))
        for cand in (reg_n, buy_n):
            if cand and (cand in target or target in cand):
                candidates[r.get("Code")] = None
                break
    if len(candidates) == 1:
        return next(iter(candidates))

    return "N/A"


# ===========================================================================
# PURE PIPELINE  (unit-tested on the dev machine with a stub model_call)
# ===========================================================================
# The system prompt encodes every extraction rule from the v2.1 spec. It demands
# JSON-only output with exact key names. The ACTIVE customer table is injected
# into the USER message each request, so the prompt always mirrors live config.
BOL_SYSTEM = """You are a meticulous Bill of Lading (B/L) data extractor for the \
Japanese market.
You parse the provided Bill of Lading document text and return ONLY a JSON object
with the extracted shipment data. No commentary, no explanations, no markdown.

OUTPUT ONLY THIS JSON OBJECT (exact key names shown; do not add other keys):
{
  "AssistantVersion": "J2.1",
  "BLNumber": "string",
  "BLDate": { "Month": integer, "Day": integer, "Year": integer },
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
  "Cartons": integer
}

Field rules:
- AssistantVersion: ALWAYS output exactly "J2.1". Never copy any version found
  in the document itself.
- BLNumber: the unique identifier of the Bill of Lading, made up ONLY of letters
  and numbers (no spaces or symbols).
- BLDate: the issue date of the B/L, split into integers Month (1-12),
  Day (1-31), Year (4 digits). If a part cannot be determined, use 0.
- CustomerName: the name of the customer receiving the shipment, copied
  VERBATIM from the document.
- CustomerCode: resolved from the REGISTERED CUSTOMER TABLE supplied in the
  user message. Match the document's customer name against the
  "RegisteredCustomerName" or "Buyer" columns (case, punctuation, brackets and
  abbreviations may vary — treat them as equivalent). If no row matches, or the
  match is ambiguous between two different codes, output "N/A". NEVER guess.
- CustomerAddress: the address of the customer receiving the shipment.
- Shipper: the name of the entity shipping the goods.
- ShipperAddress: the address of the shipping entity.
- ShipToDestination: where the shipment is delivered — City and Country as two
  separate keys.
- ShipVia: the name of the ocean vessel transporting the goods.
- VoyageNumber: the voyage number of the vessel. If no definite voyage number
  can be determined, output "N/A".
- Brand: the brand name of the goods as listed under Marks & Numbers.
- PortOfOrigin: the port where the shipment originated.
- Cartons: the TOTAL number of cartons in the shipment, as an integer. If it
  cannot be determined, use 0.

Missing fields: use "N/A" for string fields, 0 for numeric fields, and keep the
nested objects present with their inner keys defaulted ("N/A" strings / 0
numbers). NEVER invent data that is not in the document.

Return ONLY the JSON object above. No preamble, no code fences, no commentary.
"""


def build_prompt(bol_text: str, customer_table: list[dict]) -> list[dict]:
    """Assemble the OpenAI-style chat messages for a B/L extract request."""
    table_lines = "\n".join(
        f'- Code={r.get("Code")} | Buyer={r.get("Buyer")} | '
        f'RegisteredCustomerName={r.get("RegisteredCustomerName")}'
        for r in customer_table
    )
    user = (
        "Extract the following Bill of Lading. Output ONLY the required JSON "
        "object.\n\n"
        "REGISTERED CUSTOMER TABLE:\n"
        f"{table_lines}\n\n"
        "BOL_DATA:\n"
        f"<<<BOL_DATA>>>\n{bol_text}\n<<<END_BOL_DATA>>>"
    )
    return [
        {"role": "system", "content": BOL_SYSTEM},
        {"role": "user", "content": user},
    ]


# --- Thinking-suppression degradation chain -------------------------------
def _base_params() -> dict:
    """Fixed sampling parameters (independent of degradation level)."""
    return {
        "temperature": MODEL_TEMP,
        "top_p": MODEL_TOP_P,
        "top_k": MODEL_TOP_K,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": MODEL_MAX_TOKENS,
    }


def build_params(level: int = 0) -> dict:
    """Full (level-0) sampling parameters with thinking suppression.

    Level 0 includes both ``reasoning_effort`` (if DISABLE_THINKING) and
    ``chat_template_kwargs: {enable_thinking: false}``. Higher levels are
    derived by :func:`_params_at_level` (dropping fields the server rejected).
    """
    params = _base_params()
    if not DISABLE_THINKING:
        return params
    params["reasoning_effort"] = 0
    if level == 0:
        params["chat_template_kwargs"] = {"enable_thinking": False}
    return params


def _params_at_level(params: dict, level: int) -> dict:
    """Strip suppressed fields progressively as the degradation level rises."""
    out = dict(params)
    if level >= 1:
        out.pop("chat_template_kwargs", None)
    if level >= 2:
        out.pop("reasoning_effort", None)
    return out


# --- Client-side context guard --------------------------------------------
def estimate_tokens(text: str) -> int:
    """Conservative token heuristic (chars/3, ceiling). Dense B/L text runs
    low, so erring high avoids false rejections."""
    if not text:
        return 0
    return (len(str(text)) + 2) // 3


def _estimate_prompt_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def slot_budget() -> int:
    """Tokens available for a single request's prompt in one slot.

    llama.cpp divides its total ``--ctx-size`` across ``--parallel`` slots, so
    the per-slot budget is ``CONTEXT_SIZE // MODEL_PARALLEL``.
    """
    return max(1, CONTEXT_SIZE // MODEL_PARALLEL)


def usable_prompt_room() -> int:
    """Slot budget minus reserved output headroom minus safety margin."""
    return max(0, slot_budget() - MODEL_MAX_TOKENS - _SAFETY_MARGIN)


def check_context(messages: list[dict]) -> None:
    """Reject before any network call if the prompt cannot fit in a slot.

    Honors CONTEXT_GUARD: ``strict`` raises ``ContextGuardExceeded`` (422),
    ``warn`` logs a warning and allows, ``off`` skips entirely.
    """
    room = usable_prompt_room()
    est = _estimate_prompt_tokens(messages)
    if est <= room:
        return
    if CONTEXT_GUARD == "off":
        return
    if CONTEXT_GUARD == "warn":
        logger.warning(
            "Estimated prompt size %d exceeds slot budget %d; allowing request anyway.",
            est, room,
        )
        return
    raise ContextGuardExceeded(
            f"Estimated prompt size {est} tokens exceeds the gateway slot budget "
            f"{room} tokens (CONTEXT_SIZE={CONTEXT_SIZE}, MODEL_PARALLEL={MODEL_PARALLEL}, "
            f"max_tokens={MODEL_MAX_TOKENS}, safety margin {_SAFETY_MARGIN}). "
            f"Per-token slot budget = CONTEXT_SIZE // MODEL_PARALLEL = {slot_budget()} tokens. "
            f"To proceed on the DGX: re-provision the shared llama-server with a larger "
            f"--ctx-size (e.g. 131072) and lower --parallel (e.g. 2), then set "
            f"CONTEXT_SIZE={CONTEXT_SIZE} and MODEL_PARALLEL={MODEL_PARALLEL} in .env to match "
            f"and restart the gateway. Or submit a shorter B/L document."
        )


# ---------------------------------------------------------------------------
# JSON output extraction / sanitization
# ---------------------------------------------------------------------------
# Markers that indicate leaked reasoning/thinking content.
_THINKING_BLOCKS = [
    (r"<\|begin_thinking\|>.*?<\|end_thinking\|>", ""),
    (r"<think>.*?</think>", ""),
    (r"<reasoning>.*?</reasoning", ""),
]
_THINKING_MARKERS = re.compile(
    r"<\|begin_thinking\|>|<\|end_thinking\|>|<think>|</think>|<reasoning>|</reasoning>"
)


def _to_int(value: Any) -> int:
    """Coerce to int where possible; never raise (returns 0 otherwise)."""
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return 0


def _field_str(value: Any, default: str = "N/A") -> str:
    """Return a non-empty string, else the default."""
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _normalize_bl_date(value: Any) -> dict:
    if not isinstance(value, dict):
        value = {}
    return {
        "Month": _to_int(value.get("Month")),
        "Day": _to_int(value.get("Day")),
        "Year": _to_int(value.get("Year")),
    }


def _normalize_ship_to(value: Any) -> dict:
    if not isinstance(value, dict):
        value = {}
    return {
        "City": _field_str(value.get("City")),
        "Country": _field_str(value.get("Country")),
    }


def _normalize_bol(data: Any) -> dict:
    """Coerce parsed JSON into the exact J2.1 shipment schema.

    - Drops any keys not in the schema (spelling-exact keys only).
    - Coerces BLDate components and Cartons to int (non-numeric -> 0).
    - Missing/unset string fields default to "N/A" (spec §6 placeholders).
    - Forces AssistantVersion to "J2.1" regardless of model output.
    - NEVER invents data.
    """
    if not isinstance(data, dict):
        raise ModelUnavailable("model output is not a JSON object")

    out: dict[str, Any] = {
        "AssistantVersion": VERSION,
        "BLNumber": _field_str(data.get("BLNumber")),
        "BLDate": _normalize_bl_date(data.get("BLDate")),
        "CustomerName": _field_str(data.get("CustomerName")),
        "CustomerCode": _field_str(data.get("CustomerCode")),
        "CustomerAddress": _field_str(data.get("CustomerAddress")),
        "Shipper": _field_str(data.get("Shipper")),
        "ShipperAddress": _field_str(data.get("ShipperAddress")),
        "ShipToDestination": _normalize_ship_to(data.get("ShipToDestination")),
        "ShipVia": _field_str(data.get("ShipVia")),
        "VoyageNumber": _field_str(data.get("VoyageNumber")),
        "Brand": _field_str(data.get("Brand")),
        "PortOfOrigin": _field_str(data.get("PortOfOrigin")),
        "Cartons": _to_int(data.get("Cartons")),
    }
    return out


def extract_json(raw: Optional[str], customer_table: list[dict]) -> dict:
    """Parse the model's raw text into the shipment schema.

    - Strips thinking/reasoning blocks (defense-in-depth against thinking mode).
    - Strips code fences (```` ```json ... ``` ````) and surrounding whitespace.
    - Locates the JSON span (first `{` to last `}`) and json.loads it.
    - Re-resolves CustomerCode against the LIVE config table (the model's
      resolution is advisory; the gateway's lookup is authoritative).
    - Normalizes to the exact schema.
    - FAILS CLOSED (raises ModelUnavailable) if nothing parseable is found —
      a fabricated or malformed shipment record is dangerous downstream.
    """
    if raw is None:
        raise ModelUnavailable("model returned no output")

    text = str(raw)

    # Remove thinking/reasoning blocks (DOTALL so it spans newlines).
    for pattern, repl in _THINKING_BLOCKS:
        text = re.sub(pattern, repl, text, flags=re.DOTALL)
    text = _THINKING_MARKERS.sub("", text)

    # Strip JSON code fences: ```json ... ``` or ``` ... ```.
    text = re.sub(r"```(?:[A-Za-z]+)\s*", "", text)
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            raise ModelUnavailable("model output contained a JSON span but failed to parse")
        record = _normalize_bol(data)
        # Gateway-side authoritative customer-code resolution (config-driven).
        record["CustomerCode"] = lookup_customer(record.get("CustomerName"), customer_table)
        return record

    raise ModelUnavailable("could not find a JSON object in model output")


# ===========================================================================
# EXTRACT ORCHESTRATION  (pure; the only part that touches the model)
# ===========================================================================
def run_extract(request: dict, customer_table: list[dict],
                model_call: Callable[[list, dict], str]) -> dict:
    """Run a single extract request against ``model_call`` and return a
    normalized shipment record.

    ``model_call(messages, params) -> str`` is injected so this is pure/testable.
    """
    bol_text = request.get("bol_text")
    if bol_text is None:
        raise ValueError("Missing 'bol_text' field.")
    if not isinstance(bol_text, str):
        raise ValueError("'bol_text' must be a string.")
    if not bol_text.strip():
        raise ValueError("'bol_text' must be a non-empty string.")

    messages = build_prompt(bol_text, customer_table)

    # Reject locally (before any network call) if the prompt cannot fit in a slot.
    check_context(messages)

    params = build_params(_PARAMS_LEVEL)
    raw = model_call(messages, params)
    record = extract_json(raw, customer_table)
    logger.info(
        f"extract complete | bl_number={record.get('BLNumber')} "
        f"customer_code={record.get('CustomerCode')} | keys={len(record)}"
    )
    return record


def run_extract_batch(entries: list, customer_table: list[dict],
                      model_call: Callable[[list, dict], str]) -> list:
    """Run many extract entries SEQUENTIALLY, isolating per-entry failures.

    Each entry gets its own model call so one bad input cannot poison the
    others or overflow the shared slot budget. Overall HTTP status stays 200;
    each item reports ``status: "ok"`` or ``status: "error"``.
    """
    results: list[dict] = []
    for entry in entries:
        eid = entry.get("id") if isinstance(entry, dict) else None
        payload = {"bol_text": entry.get("bol_text")} if isinstance(entry, dict) else {"bol_text": None}
        try:
            record = run_extract(payload, customer_table, model_call)
            results.append({"id": eid, "status": "ok", "data": record})
        except Exception as exc:  # noqa: BLE001 - record per-entry error, never crash the batch
            logger.error("extract entry %s failed: %s", eid, exc)
            results.append({"id": eid, "status": "error", "error": str(exc)})
    return results


# ===========================================================================
# MODEL BACKEND  (HTTP to the local llama.cpp server — used on the DGX)
# ===========================================================================
def _resolve_content(message: dict) -> str:
    """Return the model's text from the first non-empty field.

    Qwen3.8-27B is an adaptive thinking model: for hard inputs it may leave
    ``content`` empty and put the answer in ``reasoning_content`` (or a legacy
    name). We fall back across those fields, logging a warning when a reasoning
    field actually carried the text (thinking leaked despite suppression).
    """
    content = message.get("content") or ""
    if content:
        return content
    for field in ("reasoning_content", "thinking", "reasoning", "reason"):
        value = message.get(field)
        if value:
            logger.warning(f"Model answered in '{field}' (thinking leaked); extracting.")
            return str(value)
    raise ModelUnavailable("Model returned an empty response.")


def _post(body: dict, headers: dict) -> requests.Response:
    try:
        return requests.post(MODEL_URL, json=body, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        # Connection refused / timeout / DNS etc. → model is unavailable (503).
        raise ModelUnavailable(f"Model server unreachable: {exc}") from exc


def http_model_call(messages: list[dict], params: dict = None) -> str:
    """POST a chat completion to the llama.cpp OpenAI-compatible endpoint.

    Implements the thinking-suppression degradation chain: try full params
    (level 0), then drop chat_template_kwargs (level 1), then drop
    reasoning_effort too (level 2). The first working level is cached so later
    calls skip straight to it. Context-length rejections map to
    ``ContextExceeded`` (503); the resolved text is returned as-is for
    ``extract_json()`` to clean.
    """
    global _PARAMS_LEVEL

    if params is None:
        params = {}
    headers = {"Authorization": f"Bearer {LLAMA_SERVER_API_KEY}"} if LLAMA_SERVER_API_KEY else {}

    level = _PARAMS_LEVEL
    while level <= 2:
        attempt = _params_at_level(params, level)
        body = {"model": MODEL_NAME, "messages": messages}
        for key in _SAMPLING_KEYS:
            if key in attempt:
                body[key] = attempt[key]
        if "reasoning_effort" in attempt:
            body["reasoning_effort"] = attempt["reasoning_effort"]
        if "chat_template_kwargs" in attempt:
            body["chat_template_kwargs"] = attempt["chat_template_kwargs"]

        resp = _post(body, headers)
        if resp.status_code == 200:
            _PARAMS_LEVEL = level
            return _resolve_content(resp.json()["choices"][0]["message"])

        detail = resp.text[:300]
        lowered = detail.lower()

        # Context-length overflow → actionable 503.
        if "context length" in lowered or ("context" in lowered and any(
            w in lowered for w in ("exceed", "size", "length", "overflow", "too long"))):
            raise ContextExceeded(
                "Request exceeds the model's context window (server ctx is fixed at launch). "
                "Restart llama-server with a larger --ctx-size and set CONTEXT_SIZE in .env to match."
            )

        # Thinking-suppression degradation: the build rejected a suppression field.
        if "chat_template_kwargs" in lowered or "reasoning_effort" in lowered:
            level += 1
            continue

        # Any other non-200 → give up cleanly.
        raise ModelUnavailable(f"Model server responded {resp.status_code}: {detail}")

    raise ModelUnavailable(
        "Model server could not satisfy the request after param degradation."
    )


# ===========================================================================
# FastAPI Application
# ===========================================================================
app = FastAPI(
    title="AI MDPL Bill of Lading Extractor API",
    description=(
        "Lean JSON-in / JSON-out Bill of Lading Extractor for the Japan market "
        "(assistant version J2.1). Runs Qwen3.8-27B locally on the shared NVIDIA "
        "DGX Spark model server."
    ),
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Lean API-key authentication -------------------------------------------
def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias=API_KEY_AUTH_HEADER, description="API key for access control"),
):
    # Optional header so a *missing* key also yields 401 (not 422).
    if not x_api_key or x_api_key not in API_KEY_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return x_api_key


# --- Request models ----------------------------------------------------------
class ExtractRequest(BaseModel):
    bol_text: str = Field(..., min_length=1, description="Pre-extracted Bill of Lading document text to parse.")


class BatchEntry(BaseModel):
    id: Optional[str] = Field(default=None, description="Client-supplied id echoed back in the result item.")
    bol_text: str = Field(..., min_length=1, description="Pre-extracted Bill of Lading document text to parse.")


class BatchRequest(BaseModel):
    entries: list[BatchEntry] = Field(
        ..., min_length=1, description="One or more B/L entries to extract (processed sequentially)."
    )


# --- Endpoints --------------------------------------------------------------
@app.get("/", tags=["System"])
async def root():
    return {
        "service": "AI MDPL Bill of Lading Extractor",
        "version": VERSION,
        "endpoints": ["/v1/extract", "/v1/extract_batch", "/v1/version", "/healthz"],
        "docs": "/docs",
    }


@app.get("/healthz", tags=["System"])
async def health_check():
    """Liveness probe — verifies model-server connectivity and reports the
    computed per-slot context budget, cross-checked against the server's
    per-slot n_ctx via /props when available."""
    model_status = "unreachable"
    server_ctx_size = None
    try:
        health_url = MODEL_URL.replace("/v1/chat/completions", "/health")
        r = requests.get(health_url, timeout=5)
        if r.status_code == 200:
            model_status = "ready"
            try:
                payload = r.json()
                server_ctx_size = payload.get("ctx_size") or payload.get("n_ctx")
            except Exception:  # noqa: BLE001 - health payload is best-effort
                server_ctx_size = None
        else:
            model_status = f"responding (status {r.status_code})"
    except Exception as exc:  # noqa: BLE001 - report any connection issue
        model_status = f"unreachable ({exc})"

    # Cross-check the gateway's assumed slot budget against the server's.
    slot = slot_budget()
    ctx_check = "unknown"
    try:
        props_url = MODEL_URL.replace("/v1/chat/completions", "/props")
        rp = requests.get(props_url, timeout=5)
        if rp.status_code == 200:
            server_slot = (
                rp.json().get("default_generation_settings", {}).get("n_ctx")
            )
            if server_slot is not None:
                ctx_check = "match" if server_slot == slot else "mismatch"
    except Exception:  # noqa: BLE001 - /props is best-effort (depends on build)
        pass

    return {
        "status": "healthy" if model_status == "ready" else "degraded",
        "model_server": model_status,
        "model_name": MODEL_NAME,
        "context_budget": {
            "slot_tokens": slot,
            "server_ctx_size": server_ctx_size,
            "check": ctx_check,
        },
        "customer_table_rows": len(load_customer_table()),
        "version": VERSION,
    }


@app.post("/v1/extract", response_class=JSONResponse, tags=["Extract"])
async def extract(
    _api_key: str = Depends(verify_api_key),
    body: ExtractRequest = Body(...),
):
    """Parse a single B/L document into the J2.1 shipment JSON."""
    table = load_customer_table()
    try:
        record = run_extract(body.model_dump(), table, http_model_call)
    except ValueError as exc:
        # Includes ContextGuardExceeded (422 with actionable sizing guidance).
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except (ModelUnavailable, ContextExceeded) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - avoid leaking internals
        logger.error(f"Unexpected extract error: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Processing failed.")

    return JSONResponse(content=record)


@app.post("/v1/extract_batch", response_class=JSONResponse, tags=["Extract"])
async def extract_batch(
    _api_key: str = Depends(verify_api_key),
    body: BatchRequest = Body(...),
):
    """Parse many B/L documents sequentially. Per-entry failures are isolated
    and reported per item; overall status stays 200."""
    table = load_customer_table()
    try:
        results = run_extract_batch(
            [entry.model_dump() for entry in body.entries],
            table,
            http_model_call,
        )
    except Exception as exc:  # noqa: BLE001 - avoid leaking internals
        logger.error(f"Unexpected batch error: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Processing failed.")

    return JSONResponse(content={"results": results})


@app.post("/v1/version", response_class=JSONResponse, tags=["System"])
async def version():
    """Service version (J2.1) — never touches the model."""
    return JSONResponse(content={"version": VERSION})


# ===========================================================================
# Standalone Runner
# ===========================================================================
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting AI MDPL Bill of Lading Extractor API on {API_HOST}:{API_PORT}")
    logger.info(f"Model server: {MODEL_URL} ({MODEL_NAME})")
    logger.info(f"DISABLE_THINKING={DISABLE_THINKING} | temp={MODEL_TEMP} | top_p={MODEL_TOP_P}")
    logger.info(
        f"CONTEXT_SIZE={CONTEXT_SIZE} MODEL_PARALLEL={MODEL_PARALLEL} | slot budget="
        f"{slot_budget()} tokens | prompt room={usable_prompt_room()} | guard={CONTEXT_GUARD}"
    )
    logger.info(f"Customer table rows loaded: {len(load_customer_table())}")
    logger.info(f"Swagger UI: http://{API_HOST}:{API_PORT}/docs")

    uvicorn.run("bol_service:app", host=API_HOST, port=API_PORT, log_level="info")
