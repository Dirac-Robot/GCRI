#!/bin/bash
# GCRI BigCodeBench Benchmark Runner
# Usage: ./run_bigcodebench.sh [ato_args...] [inspect_args...]
# Example: ./run_bigcodebench.sh agents.strategy_generator.model_id=openai/gpt-4o --limit 10

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SERVER_PORT=8001
SERVER_URL="http://localhost:$SERVER_PORT"
LOG_DIR="$PROJECT_DIR/logs"
SERVER_LOG="$LOG_DIR/gcri_server.log"
mkdir -p "$LOG_DIR"

ATO_ARGS=()
INSPECT_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --* ]]; then
        INSPECT_ARGS+=("$arg")
    else
        ATO_ARGS+=("$arg")
    fi
done

check_server() {
    curl -s "$SERVER_URL/health" > /dev/null 2>&1
}

if ! check_server; then
    echo "🚀 Starting GCRI Benchmark Server..."
    echo "   Server log: $SERVER_LOG"
    nohup env PYTHONUNBUFFERED=1 python -m gcri.benchmark.server benchmark_mode "${ATO_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
    disown
    for i in {1..30}; do
        if check_server; then echo "✅ Server ready (persistent)"; break; fi
        sleep 1
    done
    if ! check_server; then
        echo "❌ Server failed to start"; exit 1
    fi
else
    echo "✅ Server already running"
fi

echo ""
echo "🧪 Running BigCodeBench Benchmark..."
echo "═══════════════════════════════════════════════════════════"
PYTHONPATH=. inspect eval gcri/benchmark/bigcodebench.py "${INSPECT_ARGS[@]}"
