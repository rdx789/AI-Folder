#!/usr/bin/env bash
# Usage:  ./setup.sh   (or: bash setup.sh)
#         source setup.sh   also works from bash or zsh: setup runs in a separate bash
#         (its strict error settings never reach your shell), then .venv is activated here.
if [ -n "${ZSH_EVAL_CONTEXT:-}" ]; then
  case "$ZSH_EVAL_CONTEXT" in *:file*) MAYA_SETUP_SOURCED="$0" ;; esac   # zsh: $0 is the sourced file
elif [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; then
  MAYA_SETUP_SOURCED="${BASH_SOURCE[0]}"
fi
if [ -n "${MAYA_SETUP_SOURCED:-}" ]; then
  MAYA_SETUP_DIR="$(cd -- "$(dirname -- "$MAYA_SETUP_SOURCED")" && pwd)"
  unset MAYA_SETUP_SOURCED
  if bash "$MAYA_SETUP_DIR/setup.sh"; then
    . "$MAYA_SETUP_DIR/.venv/bin/activate"
    echo "Activated $MAYA_SETUP_DIR/.venv in this shell."
    unset MAYA_SETUP_DIR
    return 0
  fi
  echo "setup.sh failed (see above); your shell settings were not changed." >&2
  unset MAYA_SETUP_DIR
  return 1
fi

set -euo pipefail
MAYA_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$MAYA_PROJECT_DIR"
MAYA_PYTHON="${MAYA_PYTHON:-python3}"
"$MAYA_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Maya requires Python 3.10 or later"'
if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
fi
if [[ ! -d .venv ]]; then
  "$MAYA_PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install -e .
if [[ -d code/tests ]]; then
  .venv/bin/python -m unittest discover -s code/tests -q
else
  printf 'code/tests/ not present (tests are not in git); skipping unit tests.\n'
fi
.venv/bin/python -m maya.ingest
printf '\nSetup complete. Activate with: source .venv/bin/activate\n'
printf 'Run ingestion validation: maya-ingest\n'
printf 'Offline S2 replay: maya-eval (reports go to results/)\n'
