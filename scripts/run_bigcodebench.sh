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

# Auto-split arguments by format:
#   key=value or single_word (no dash) -> ato
#   --key or -k -> inspect
ATO_ARGS=()
INSPECT_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --* ]] || [[ "$arg" == -* && ${#arg} -eq 2 ]]; then
        INSPECT_ARGS+=("$arg")
    elif [[ "$arg" == *=* ]] || [[ "$arg" != -* ]]; then
        ATO_ARGS+=("$arg")
    else
        INSPECT_ARGS+=("$arg")
    fi
done

check_server() {
    curl -s "$SERVER_URL/health" > /dev/null 2>&1
}

if ! check_server; then
    echo "🚀 Starting GCRI Benchmark Server..."
    source .venv/bin/activate
    python -m gcri.benchmark.server benchmark_mode "${ATO_ARGS[@]}" &
    SERVER_PID=$!
    for i in {1..30}; do
        if check_server; then echo "✅ Server ready"; break; fi
        sleep 1
    done
    if ! check_server; then
        echo "❌ Server failed"; kill $SERVER_PID 2>/dev/null || true; exit 1
    fi
else
    echo "✅ Server already running"
fi

echo "🧪 Running BigCodeBench Benchmark..."
source .venv/bin/activate
PYTHONPATH=. inspect eval gcri/benchmark/bigcodebench.py "${INSPECT_ARGS[@]}"
