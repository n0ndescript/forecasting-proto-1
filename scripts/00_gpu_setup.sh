#!/usr/bin/env bash
# One-shot setup for a fresh CUDA host (RunPod H100/A100 etc.).
# Run from the project root after cloning/transferring the repo:
#
#   bash scripts/00_gpu_setup.sh
#
# What it does:
#   1. Confirms CUDA is visible.
#   2. Installs uv if not already present.
#   3. Runs `uv sync --extra gpu` (pulls torch, earth2studio, etc.).
#   4. Installs flash_attn (prebuilt wheel; falls back to source build).
#   5. Smoke-tests that torch+CUDA work and that AIFS+ARCO import cleanly.
#
# After it succeeds, the next step is:
#   uv run python scripts/02_download_one_date.py   # B2 verification
#   uv run python scripts/04_run_forecasts.py       # B3 full batch

set -eu   # NOT pipefail — `nvidia-smi | head` triggers SIGPIPE which would kill us

banner() { echo; echo "=========================================="; echo "$1"; echo "=========================================="; }

banner "1. nvidia-smi"
nvidia-smi || true   # informational; don't die if missing

banner "2. install uv (if missing)"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

banner "3. uv sync --extra gpu"
uv sync --extra gpu

banner "4. install flash_attn"
# Try prebuilt wheel first; if not available for this torch/CUDA combo,
# fall back to a source build (~10 min on H100). NOTE: uv pip install
# returns 0 even when it builds from source, so we can't easily tell
# which path was taken — that's fine, both end with flash_attn working.
uv pip install flash-attn --no-build-isolation

banner "5. smoke tests"
uv run python - <<'PY'
import torch
print(f"torch:          {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print(f"VRAM:           {p.total_memory / 1024**3:.1f} GB")
    print(f"compute cap:    sm_{p.major}{p.minor}")

print()
print("--- earth2studio imports ---")
from earth2studio.models.px import AIFS
from earth2studio.data import ARCO
print("AIFS + ARCO: OK")

print()
print("--- flash_attn ---")
try:
    import flash_attn
    print(f"flash_attn:     {flash_attn.__version__}")
except Exception as e:
    print(f"flash_attn:     FAILED: {e}")
PY

banner "DONE. Next step:"
echo "  uv run python scripts/02_download_one_date.py   # B2 verification (one date end-to-end)"
echo "  uv run python scripts/04_run_forecasts.py       # B3 full 122-forecast batch"
