#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"
runtime_root="$repo_root/.runtime"
source_root="$runtime_root/sources/mmagic-v1.2.0"
checkpoint_root="$runtime_root/checkpoints/mmagic-realbasicvsr"
environment_root="$runtime_root/envs/mmagic-realbasicvsr"
python_install_root="$runtime_root/python"
uv_cache_root="$runtime_root/uv-cache"

source_url="https://github.com/open-mmlab/mmagic.git"
source_commit="c749dcc7172d198ac2a27c3e5a4d2181640f0fd5"
checkpoint_url="https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth"
checkpoint_sha256="52f77c2c835aaa3fe675b3959b2f85010a6c6f63f77f7e279394646e55a4e376"
checkpoint_size_bytes="148239017"
checkpoint_path="$checkpoint_root/RealBasicVSR.pth"

uv_binary="$repo_root/.venv/bin/uv"
if [[ ! -x "$uv_binary" ]]; then
    echo "missing uv: run .venv/bin/pip install -e '.[dev]'" >&2
    exit 1
fi

mkdir -p "$runtime_root/sources" "$checkpoint_root" "$uv_cache_root"

if [[ -d "$source_root/.git" ]]; then
    actual_commit="$(git -C "$source_root" rev-parse HEAD)"
    if [[ "$actual_commit" != "$source_commit" ]]; then
        echo "MMagic source commit mismatch: expected=$source_commit actual=$actual_commit" >&2
        exit 1
    fi
else
    if [[ -e "$source_root" ]]; then
        echo "MMagic source path exists but is not a Git checkout: $source_root" >&2
        exit 1
    fi
    source_partial="$source_root.partial"
    if [[ -e "$source_partial" ]]; then
        echo "stale partial source checkout exists: $source_partial" >&2
        exit 1
    fi
    git clone "$source_url" "$source_partial"
    git -C "$source_partial" checkout --detach "$source_commit"
    mv -- "$source_partial" "$source_root"
fi

if [[ -f "$checkpoint_path" ]]; then
    actual_size="$(stat -c '%s' "$checkpoint_path")"
    actual_sha256="$(sha256sum "$checkpoint_path" | cut -d ' ' -f 1)"
    if [[ "$actual_size" != "$checkpoint_size_bytes" || "$actual_sha256" != "$checkpoint_sha256" ]]; then
        echo "existing RealBasicVSR checkpoint failed validation: $checkpoint_path" >&2
        exit 1
    fi
else
    checkpoint_partial="$checkpoint_path.partial"
    curl --location --fail --retry 5 --continue-at - --output "$checkpoint_partial" "$checkpoint_url"
    actual_size="$(stat -c '%s' "$checkpoint_partial")"
    if [[ "$actual_size" != "$checkpoint_size_bytes" ]]; then
        echo "checkpoint size mismatch: expected=$checkpoint_size_bytes actual=$actual_size" >&2
        exit 1
    fi
    actual_sha256="$(sha256sum "$checkpoint_partial" | cut -d ' ' -f 1)"
    if [[ "$actual_sha256" != "$checkpoint_sha256" ]]; then
        echo "checkpoint SHA-256 mismatch: expected=$checkpoint_sha256 actual=$actual_sha256" >&2
        exit 1
    fi
    mv -- "$checkpoint_partial" "$checkpoint_path"
fi

export UV_CACHE_DIR="$uv_cache_root"
export UV_PYTHON_INSTALL_DIR="$python_install_root"
"$uv_binary" python install 3.11.13
if [[ ! -x "$environment_root/bin/python" ]]; then
    "$uv_binary" venv --python 3.11.13 "$environment_root"
fi

actual_python="$("$environment_root/bin/python" -c 'import platform; print(platform.python_version())')"
if [[ "$actual_python" != "3.11.13" ]]; then
    echo "RealBasicVSR worker Python mismatch: expected=3.11.13 actual=$actual_python" >&2
    exit 1
fi

"$uv_binary" pip install \
    --python "$environment_root/bin/python" \
    --default-index https://pypi.org/simple \
    --index https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match \
    'torch==2.6.0+cu124'
"$uv_binary" pip install \
    --python "$environment_root/bin/python" \
    'numpy==1.26.4'

"$environment_root/bin/python" - <<'PY'
import torch

if torch.__version__ != "2.6.0+cu124":
    raise RuntimeError(f"unexpected torch version: {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the RealBasicVSR worker environment")
if torch.cuda.get_device_capability(0) != (8, 6):
    raise RuntimeError(
        f"unexpected GPU compute capability: {torch.cuda.get_device_capability(0)}"
    )
print(
    "RealBasicVSR runtime ready:",
    torch.__version__,
    torch.cuda.get_device_name(0),
    torch.cuda.get_device_capability(0),
)
PY

PYTHONPATH="$repo_root/src" \
    "$environment_root/bin/python" "$repo_root/scripts/smoke_realbasicvsr.py"
