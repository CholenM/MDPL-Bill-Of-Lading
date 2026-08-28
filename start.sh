# ===========================================================================
# AI MDPL Bill of Lading Extractor — Launch Script (DGX Spark)
# ===========================================================================
# The model server (vLLM) is ALREADY RUNNING on the DGX. This script
# only launches the FastAPI GATEWAY and connects to that model. It does NOT
# start, stop, or kill the model server.
#
# Lightweight bootstrap: if the venv or .env don't exist yet it creates them,
# so ./start.sh remains a single entry point.
#
# Usage: ./start.sh
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ---------------------------------------------------------------------------
# Step 1: Bootstrap venv (one-time)
# ---------------------------------------------------------------------------
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo -e "${CYAN}[1/3] Creating Python virtual environment...${NC}"
    python3 -m venv "$SCRIPT_DIR/.venv"
fi
source "$SCRIPT_DIR/.venv/bin/activate"

if ! python -c "import fastapi, requests, dotenv, httpx" >/dev/null 2>&1; then
    echo -e "${CYAN}[1/3] Installing runtime dependencies...${NC}"
    pip install --upgrade pip -q
    pip install -r "$SCRIPT_DIR/requirements.txt" -q
fi

# ---------------------------------------------------------------------------
# Step 2: Bootstrap .env (one-time) from the example
# ---------------------------------------------------------------------------
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo -e "${CYAN}[2/3] .env not found — creating from .env.example${NC}"
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo -e "${YELLOW}  .env generated. Review it (API keys, model URL, ports) before continuing.${NC}"
fi

# Load .env
set -a; source .env; set +a

MODEL_URL="${MODEL_URL:-http://127.0.0.1:8011/v1/chat/completions}"
# Shared model server port, parsed from MODEL_URL (the single source of truth).
# Defaults to 8011 if MODEL_URL has no :port. Kept defined so `set -u` is happy.
_MODEL_PORT="${MODEL_URL#*://}"; _MODEL_PORT="${_MODEL_PORT%%/*}"; _MODEL_PORT="${_MODEL_PORT##*:}"
MODEL_PORT="${_MODEL_PORT//[!0-9]/}"
MODEL_PORT="${MODEL_PORT:-8011}"
API_PORT="${API_PORT:-8086}"
API_HOST="${API_HOST:-0.0.0.0}"
# How long to wait for the SHARED model server (managed by DGXSpark_Setup/vllm-qwen)
# to be healthy before launching the gateway. vLLM needs up to 15 minutes on
# first boot for FlashInfer JIT kernel compilation.
SERVER_WAIT="${SERVER_WAIT:-300}"
# How long to wait for OUR gateway's own /healthz.
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-30}"

# ---------------------------------------------------------------------------
# Step 3: Pre-flight — require the shared vLLM server (:8011) to be healthy
#         (it is managed by DGXSpark_Setup/vllm-qwen/startserver.sh — we never
#         start or kill it here; this script only connects to it)
# ---------------------------------------------------------------------------
echo -e "${CYAN}[3/3] Pre-flight: shared vLLM server (timeout ${SERVER_WAIT}s)...${NC}"
MODEL_HEALTH_URL="${MODEL_URL%/v1/chat/completions}/health"
MODEL_READY="no"
ELAPSED=0
while [ $ELAPSED -lt "$SERVER_WAIT" ]; do
    # /health is unauthenticated on vLLM, so no bearer header needed.
    if curl -sf "$MODEL_HEALTH_URL" >/dev/null 2>&1; then
        MODEL_READY="yes"
        break
    fi
    sleep 2; ELAPSED=$((ELAPSED + 2))
done
if [ "$MODEL_READY" != "yes" ]; then
    echo -e "${RED}ERROR: Shared vLLM server not reachable at ${MODEL_HEALTH_URL} after ${SERVER_WAIT}s.${NC}"
    echo -e "  It is managed by DGXSpark_Setup/vllm-qwen/startserver.sh (not this project)."
    echo -e "  Start vLLM first (from DGXSpark_Setup/vllm-qwen):  ${CYAN}./startserver.sh${NC}"
    echo -e "  then run:  ${CYAN}./start.sh${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Shared vLLM server healthy on :${MODEL_PORT}"

# Also probe /v1/models as a vLLM health validation (confirms the model loaded).
MODELS_URL="${MODEL_URL%/v1/chat/completions}/v1/models"
if curl -sf "$MODELS_URL" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} vLLM /v1/models endpoint confirmed"
else
    echo -e "${YELLOW}  /v1/models not yet available (may still be loading). Continuing.${NC}"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Launch the FastAPI gateway ONLY
# ---------------------------------------------------------------------------
echo -e "${CYAN}Starting Bill of Lading Extractor gateway on :${API_PORT}...${NC}"
python3 bol_service.py &
API_PID=$!
echo "API_PID=$API_PID" > "$SCRIPT_DIR/.pids"

# Wait for our own /healthz before declaring LIVE (model already confirmed above).
echo -e "${CYAN}Waiting for gateway /healthz (up to ${HEALTH_TIMEOUT}s)...${NC}"
ELAPSED=0
while [ $ELAPSED -lt "$HEALTH_TIMEOUT" ]; do
    curl -sf "http://127.0.0.1:${API_PORT}/healthz" >/dev/null 2>&1 && { echo -e "${GREEN}✓${NC} Gateway healthy (${ELAPSED}s)"; break; }
    kill -0 "$API_PID" 2>/dev/null || { echo -e "${RED}FastAPI exited unexpectedly!${NC}"; exit 1; }
    sleep 2; ELAPSED=$((ELAPSED + 2))
done

echo ""
echo -e "${GREEN}======== Bill of Lading Extractor is LIVE ========${NC}"
echo -e "  API:    http://0.0.0.0:${API_PORT}"
echo -e "  Docs:   http://0.0.0.0:${API_PORT}/docs"
echo -e "  Health: http://0.0.0.0:${API_PORT}/healthz"
echo -e "  Model:  vLLM Qwen3.6-35B-A3B-NVFP4 on :${MODEL_PORT} (shared, managed by DGXSpark_Setup/vllm-qwen/startserver.sh)"
echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop the gateway only — the model server is left running."
echo -e "${GREEN}==================================================${NC}"

# Keep alive — watch the gateway
while true; do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo -e "${RED}Bill of Lading Extractor gateway exited unexpectedly.${NC}"
        break
    fi
    sleep 5
done

# Cleanup (trap handles SIGINT/SIGTERM); remove pid file
rm -f "$SCRIPT_DIR/.pids" 2>/dev/null || true
echo -e "${GREEN}Bill of Lading Extractor gateway stopped.${NC}"
