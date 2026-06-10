#!/usr/bin/env bash
set -euo pipefail

uv run pytest -v ./cs336_alignment/tests --junitxml=test_results.xml || true
echo "Done running tests"

output_file="cs336-spring2025-assignment-5-submission.zip"
rm "$output_file" || true

zip -r "$output_file" . \
    -x '*egg-info*' \
    -x '*mypy_cache*' \
    -x '*pytest_cache*' \
    -x '*build*' \
    -x '*ipynb_checkpoints*' \
    -x '*__pycache__*' \
    -x '*.pkl' \
    -x '*.pickle' \
    -x '*.txt' \
    -x '*.log' \
    -x '*.json' \
    -x '*.out' \
    -x '*.err' \
    -x '.git*' \
    -x '.venv/*' \
    -x '.*' \
    -x 'data/*' \
    -x 'models/*' \
    -x 'outputs/*' \
    -x 'logs/*' \
    -x '*.pt' \
    -x '*.pth' \
    -x '*.safetensors' \
    -x '*.npy' \
    -x '*.npz'

echo "All files have been compressed into $output_file"
