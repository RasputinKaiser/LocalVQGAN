#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

find_python() {
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if [ ! -x .venv/bin/python ]; then
  PYTHON=$(find_python) || {
    echo "LocalVQGAN needs Python 3.11 or newer, and no matching python3 was found on your PATH." >&2
    echo "Install one from https://www.python.org/downloads/ (or 'brew install python@3.13' on macOS)," >&2
    echo "then run ./run.sh again." >&2
    exit 1
  }
  echo "Using $("$PYTHON" --version) at $(command -v "$PYTHON")"
  "$PYTHON" -m venv .venv

  EXTRAS=""
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    EXTRAS="[mlx]"
    echo "Apple Silicon detected — installing the native MLX engine alongside PyTorch."
  fi
  echo "Installing LocalVQGAN and its dependencies (downloads ~1-2 GB, mostly PyTorch — this can take a few minutes)..."
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install -e ".$EXTRAS"
fi

exec .venv/bin/localvqgan
