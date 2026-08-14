#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"
runtime_root="$repo_root/.runtime"
source_root="$runtime_root/sources/practical-rife-v4.25"
checkpoint_root="$runtime_root/checkpoints/practical-rife-v4.25"
environment_root="$runtime_root/envs/practical-rife-v4.25"
python_install_root="$runtime_root/python"
uv_cache_root="$runtime_root/uv-cache"

source_url="https://github.com/hzwer/Practical-RIFE.git"
source_commit="17d8c7a1005b37f4c97bfee04e316aaec7fdc536"
checkpoint_url="https://drive.usercontent.google.com/download?id=1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg&export=download&confirm=t"
checkpoint_sha256="e63d481b7ae5d4a4e6ad7ac5b410ff78f3bf7be3b51b2e38ca8152747abde5b4"
checkpoint_size_bytes="22919050"

uv_binary="$repo_root/.venv/bin/uv"
if [[ ! -x "$uv_binary" ]]; then
    echo "missing uv: run .venv/bin/pip install -e '.[dev]'" >&2
    exit 1
fi

mkdir -p "$runtime_root/sources" "$runtime_root/checkpoints" "$uv_cache_root"

if [[ -d "$source_root/.git" ]]; then
    actual_commit="$(git -C "$source_root" rev-parse HEAD)"
    if [[ "$actual_commit" != "$source_commit" ]]; then
        echo "RIFE source commit mismatch: expected=$source_commit actual=$actual_commit" >&2
        exit 1
    fi
else
    if [[ -e "$source_root" ]]; then
        echo "RIFE source path exists but is not a Git checkout: $source_root" >&2
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

if [[ -d "$checkpoint_root" ]]; then
    if [[ ! -f "$checkpoint_root/train_log/flownet.pkl" ]]; then
        echo "checkpoint directory is incomplete: $checkpoint_root" >&2
        exit 1
    fi
else
    if [[ -e "$checkpoint_root" ]]; then
        echo "checkpoint path exists but is not a directory: $checkpoint_root" >&2
        exit 1
    fi
    task_tmp="$(mktemp -d "$runtime_root/rife-bootstrap.XXXXXX")"
    trap 'rm -rf -- "$task_tmp"' EXIT
    archive_path="$task_tmp/RIFEv4.25.zip"
    curl --location --fail --retry 3 --output "$archive_path" "$checkpoint_url"
    actual_size="$(stat -c '%s' "$archive_path")"
    if [[ "$actual_size" != "$checkpoint_size_bytes" ]]; then
        echo "RIFE archive size mismatch: expected=$checkpoint_size_bytes actual=$actual_size" >&2
        exit 1
    fi
    actual_sha256="$(sha256sum "$archive_path" | cut -d ' ' -f 1)"
    if [[ "$actual_sha256" != "$checkpoint_sha256" ]]; then
        echo "RIFE archive SHA-256 mismatch: expected=$checkpoint_sha256 actual=$actual_sha256" >&2
        exit 1
    fi
    unzip -q "$archive_path" -d "$task_tmp/extracted"
    if [[ ! -f "$task_tmp/extracted/train_log/flownet.pkl" ]]; then
        echo "verified RIFE archive does not contain train_log/flownet.pkl" >&2
        exit 1
    fi
    mv -- "$task_tmp/extracted" "$checkpoint_root"
fi

export UV_CACHE_DIR="$uv_cache_root"
export UV_PYTHON_INSTALL_DIR="$python_install_root"
"$uv_binary" python install 3.11.13
if [[ ! -x "$environment_root/bin/python" ]]; then
    "$uv_binary" venv --python 3.11.13 "$environment_root"
fi

actual_python="$("$environment_root/bin/python" -c 'import platform; print(platform.python_version())')"
if [[ "$actual_python" != "3.11.13" ]]; then
    echo "RIFE worker Python mismatch: expected=3.11.13 actual=$actual_python" >&2
    exit 1
fi

"$uv_binary" pip install \
    --python "$environment_root/bin/python" \
    --default-index https://pypi.org/simple \
    --index https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match \
    'torch==2.6.0+cu124' \
    'torchvision==0.21.0+cu124'
"$uv_binary" pip install \
    --python "$environment_root/bin/python" \
    'numpy==1.23.5'
"$environment_root/bin/python" - <<'PY'
import torch

if torch.__version__ != "2.6.0+cu124":
    raise RuntimeError(f"unexpected torch version: {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the RIFE worker environment")
if torch.cuda.get_device_capability(0) != (8, 6):
    raise RuntimeError(
        f"unexpected GPU compute capability: {torch.cuda.get_device_capability(0)}"
    )
print(
    "RIFE runtime ready:",
    torch.__version__,
    torch.cuda.get_device_name(0),
    torch.cuda.get_device_capability(0),
)
PY

PYTHONPATH="$repo_root/src" "$environment_root/bin/python" "$repo_root/scripts/smoke_rife.py"
