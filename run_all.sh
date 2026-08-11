#!/usr/bin/env bash
# One-command entry point for the Ascend survey.
#
#   bash run_all.sh
#
# PYTHONPATH points at this directory so that sitecustomize.py is picked up by
# every process the runner spawns; that is what registers the Ascend backend
# (torch_npu) before anything tries to detect it, including the unmodified
# official auto_bench.py.
set -u

cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Triton caches compiled kernels here. Keeping it inside the bundle means a
# re-run is fast and nothing is left behind elsewhere on the machine.
export TRITON_CACHE_DIR="$PWD/.triton-cache"

# CANN puts torch_npu's shared libraries on the loader path from here. In an
# interactive shell it is usually already sourced from a profile, but the
# runner spawns non-interactive children, and a missing ASCEND_HOME_PATH shows
# up as a bare "import torch_npu failed" that looks like a broken install.
# Sourcing it is idempotent, so do it whenever the file is there.
for env_file in /usr/local/Ascend/ascend-toolkit/set_env.sh \
                "${ASCEND_HOME_PATH:-}/../set_env.sh"; do
    if [ -f "$env_file" ]; then
        # shellcheck disable=SC1090
        . "$env_file"
        echo "sourced $env_file"
        break
    fi
done

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "no python3 or python on PATH" >&2
    exit 1
fi

echo "python:     $($PY -V 2>&1)"
echo "PYTHONPATH: $PYTHONPATH"
echo

# No -e: a stage that fails is a result to report, not a reason to abort. The
# runner decides what is fatal.
"$PY" runner.py "$@"
