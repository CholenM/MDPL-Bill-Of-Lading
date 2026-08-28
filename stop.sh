#!/bin/bash
# ===========================================================================
# AI MDPL Bill of Lading Extractor — Stop Script (DGX Spark)
# ===========================================================================
# Stops ONLY the FastAPI gateway. It deliberately NEVER touches the shared
# vLLM server (Toby's pre-existing model). The name-based fallback is scoped
# strictly to bol_service.py so we can't kill the model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

PID_FILE="$SCRIPT_DIR/.pids"

if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}No .pids file found. Gateway may not be running.${NC}"
    echo "Falling back to name-based lookup (gateway only)..."
    pkill -f "bol_service.py" 2>/dev/null && echo -e "${GREEN}✓${NC} Stopped BOL Extractor gateway (by name)" || \
        echo -e "${YELLOW}  Nothing matched. Nothing to stop.${NC}"
    exit 0
fi

echo -e "${YELLOW}Stopping Bill of Lading Extractor gateway...${NC}"

# shellcheck disable=SC1091
source "$PID_FILE"

if [ -n "${API_PID:-}" ] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null
    echo -e "${GREEN}✓${NC} Gateway stopped (PID $API_PID)"
else
    echo "  Gateway not running"
fi

rm -f "$PID_FILE"
echo -e "${GREEN}BOL Extractor stopped. (The shared vLLM server was left untouched.)${NC}"
