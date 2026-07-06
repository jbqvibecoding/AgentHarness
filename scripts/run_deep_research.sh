#!/usr/bin/env bash
# Run the multi-agent deep_research pipeline.
# Usage: ./scripts/run_deep_research.sh --question "..." [--depth quick|standard|deep] [--out DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m workflows.deep_research.run "$@"
