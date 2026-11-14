#!/usr/bin/env bash
set -euo pipefail
# Explicit Python executables and destination. Existing environments are never modified.
if [ "$#" -ne 3 ]; then echo "Usage: $0 /path/python3.13 /path/python3.9 /path/new-runtime"; exit 2; fi
release_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
"$1" -m venv "$3/web"
"$3/web/bin/python" -m pip install -r "$release_root/requirements-release-web.txt"
"$2" -m venv "$3/worker"
"$3/worker/bin/python" -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r "$release_root/requirements-release-worker.txt"
echo "Set PAMK_WEB_PYTHON and PAMK_WORKER_PYTHON in your private release env."
