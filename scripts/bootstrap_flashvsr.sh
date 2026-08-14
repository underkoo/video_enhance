#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"
runtime_root="$repo_root/.runtime"
source_root="$runtime_root/sources/flashvsr-v1.1"
block_sparse_root="$runtime_root/sources/block-sparse-attention"
checkpoint_root="$runtime_root/checkpoints/flashvsr-v1.1"
environment_root="$runtime_root/envs/flashvsr-v1.1"
toolchain_root="$runtime_root/toolchains/cuda-12.4.131"
download_root="$runtime_root/downloads"
python_install_root="$runtime_root/python"
uv_cache_root="$runtime_root/uv-cache"

source_url="https://github.com/OpenImagingLab/FlashVSR.git"
source_commit="b527c6f285fb30df530f5febc8b45764a789c961"
block_sparse_url="https://github.com/mit-han-lab/Block-Sparse-Attention.git"
block_sparse_commit="49d6c39e4dc0303442cda3bb758b3925d4399c49"
cuda_nvcc_url="https://conda.anaconda.org/nvidia/label/cuda-12.4.1/linux-64/cuda-nvcc-12.4.131-0.tar.bz2"
cuda_nvcc_sha256="8342cc8ca1d82923bf46ee42f6f61e53b7b0f16cc5189573259ab9a655b78997"
cuda_nvcc_size_bytes="65674958"
weight_revision="ad1aceeac60dbd288e51acea9096b821a8703bee"
weight_root="https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/$weight_revision"

uv_binary="$repo_root/.venv/bin/uv"
if [[ ! -x "$uv_binary" ]]; then
    echo "missing uv: run .venv/bin/pip install -e '.[dev]'" >&2
    exit 1
fi

mkdir -p \
    "$runtime_root/sources" \
    "$runtime_root/checkpoints" \
    "$runtime_root/toolchains" \
    "$download_root" \
    "$uv_cache_root"

checkout_pinned_source() {
    local name="$1"
    local url="$2"
    local commit="$3"
    local destination="$4"
    if [[ -d "$destination/.git" ]]; then
        local actual_commit
        actual_commit="$(git -C "$destination" rev-parse HEAD)"
        if [[ "$actual_commit" != "$commit" ]]; then
            echo "$name source commit mismatch: expected=$commit actual=$actual_commit" >&2
            exit 1
        fi
        return
    fi
    if [[ -e "$destination" ]]; then
        echo "$name source path exists but is not a Git checkout: $destination" >&2
        exit 1
    fi
    local partial="$destination.partial"
    if [[ -e "$partial" ]]; then
        echo "stale partial source checkout exists: $partial" >&2
        exit 1
    fi
    git clone "$url" "$partial"
    git -C "$partial" checkout --detach "$commit"
    mv -- "$partial" "$destination"
}

download_verified() {
    local filename="$1"
    local url="$2"
    local expected_size="$3"
    local expected_sha256="$4"
    local destination="$5"
    if [[ -f "$destination" ]]; then
        local actual_size actual_sha256
        actual_size="$(stat -c '%s' "$destination")"
        actual_sha256="$(sha256sum "$destination" | cut -d ' ' -f 1)"
        if [[ "$actual_size" == "$expected_size" && "$actual_sha256" == "$expected_sha256" ]]; then
            echo "verified existing artifact: $filename"
            return
        fi
        echo "existing artifact failed validation: $destination" >&2
        exit 1
    fi
    local partial="$destination.partial"
    curl --location --fail --retry 5 --continue-at - --output "$partial" "$url"
    local actual_size actual_sha256
    actual_size="$(stat -c '%s' "$partial")"
    if [[ "$actual_size" != "$expected_size" ]]; then
        echo "$filename size mismatch: expected=$expected_size actual=$actual_size" >&2
        exit 1
    fi
    actual_sha256="$(sha256sum "$partial" | cut -d ' ' -f 1)"
    if [[ "$actual_sha256" != "$expected_sha256" ]]; then
        echo "$filename SHA-256 mismatch: expected=$expected_sha256 actual=$actual_sha256" >&2
        exit 1
    fi
    mv -- "$partial" "$destination"
}

checkout_pinned_source "FlashVSR" "$source_url" "$source_commit" "$source_root"
checkout_pinned_source \
    "Block Sparse Attention" \
    "$block_sparse_url" \
    "$block_sparse_commit" \
    "$block_sparse_root"
git -C "$block_sparse_root" submodule update --init --recursive csrc/cutlass

cuda_archive="$download_root/cuda-nvcc-12.4.131-0.tar.bz2"
download_verified \
    "cuda-nvcc-12.4.131-0.tar.bz2" \
    "$cuda_nvcc_url" \
    "$cuda_nvcc_size_bytes" \
    "$cuda_nvcc_sha256" \
    "$cuda_archive"
if [[ ! -x "$toolchain_root/bin/nvcc" ]]; then
    if [[ -e "$toolchain_root" ]]; then
        echo "CUDA toolchain directory is incomplete: $toolchain_root" >&2
        exit 1
    fi
    toolchain_partial="$toolchain_root.partial"
    if [[ -e "$toolchain_partial" ]]; then
        echo "stale CUDA toolchain extraction exists: $toolchain_partial" >&2
        exit 1
    fi
    mkdir -p "$toolchain_partial"
    tar -xjf "$cuda_archive" -C "$toolchain_partial"
    if [[ ! -x "$toolchain_partial/bin/nvcc" ]]; then
        echo "verified CUDA package does not contain bin/nvcc" >&2
        exit 1
    fi
    mv -- "$toolchain_partial" "$toolchain_root"
fi

mkdir -p "$checkpoint_root"
download_verified \
    "diffusion_pytorch_model_streaming_dmd.safetensors" \
    "$weight_root/diffusion_pytorch_model_streaming_dmd.safetensors" \
    "5676070392" \
    "bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be" \
    "$checkpoint_root/diffusion_pytorch_model_streaming_dmd.safetensors"
download_verified \
    "LQ_proj_in.ckpt" \
    "$weight_root/LQ_proj_in.ckpt" \
    "575694948" \
    "d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2" \
    "$checkpoint_root/LQ_proj_in.ckpt"
download_verified \
    "TCDecoder.ckpt" \
    "$weight_root/TCDecoder.ckpt" \
    "189018333" \
    "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161" \
    "$checkpoint_root/TCDecoder.ckpt"
download_verified \
    "Wan2.1_VAE.pth" \
    "$weight_root/Wan2.1_VAE.pth" \
    "507609880" \
    "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981" \
    "$checkpoint_root/Wan2.1_VAE.pth"

export UV_CACHE_DIR="$uv_cache_root"
export UV_PYTHON_INSTALL_DIR="$python_install_root"
"$uv_binary" python install 3.11.13
if [[ ! -x "$environment_root/bin/python" ]]; then
    "$uv_binary" venv --python 3.11.13 "$environment_root"
fi

actual_python="$("$environment_root/bin/python" -c 'import platform; print(platform.python_version())')"
if [[ "$actual_python" != "3.11.13" ]]; then
    echo "FlashVSR worker Python mismatch: expected=3.11.13 actual=$actual_python" >&2
    exit 1
fi

"$uv_binary" pip install \
    --python "$environment_root/bin/python" \
    --default-index https://pypi.org/simple \
    --index https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match \
    'torch==2.6.0+cu124' \
    'torchvision==0.21.0+cu124' \
    'torchaudio==2.6.0+cu124'
"$uv_binary" pip install \
    --python "$environment_root/bin/python" \
    --default-index https://pypi.org/simple \
    --index https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match \
    --requirement "$repo_root/requirements/flashvsr-worker.lock" \
    --exclude-newer '2025-08-31T00:00:00Z'

site_packages="$environment_root/lib/python3.11/site-packages"
cuda_runtime_root="$site_packages/nvidia/cuda_runtime"
cuda_cccl_root="$site_packages/nvidia/cuda_cccl"
if [[ ! -f "$cuda_runtime_root/include/cuda_runtime.h" ]]; then
    echo "CUDA runtime headers are missing: $cuda_runtime_root/include" >&2
    exit 1
fi
if [[ ! -f "$cuda_cccl_root/include/nv/target" ]]; then
    echo "CUDA CCCL headers are missing: $cuda_cccl_root/include" >&2
    exit 1
fi
if [[ ! -f "$cuda_runtime_root/lib/libcudart.so.12" ]]; then
    echo "CUDA runtime library is missing: $cuda_runtime_root/lib/libcudart.so.12" >&2
    exit 1
fi
if [[ ! -e "$toolchain_root/lib/libcudart.so" ]]; then
    ln -s -- "$cuda_runtime_root/lib/libcudart.so.12" "$toolchain_root/lib/libcudart.so"
fi

export CUDA_HOME="$toolchain_root"
cuda_include_path="$(find "$site_packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name include -print | sort | paste -sd ':' -)"
cuda_library_path="$(find "$site_packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd ':' -)"
if [[ -z "$cuda_include_path" || -z "$cuda_library_path" ]]; then
    echo "PyTorch CUDA dependency headers or libraries are missing" >&2
    exit 1
fi
export CPATH="$cuda_include_path:$toolchain_root/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$cuda_library_path${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$cuda_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$toolchain_root/bin:$PATH"
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80"
export BLOCK_SPARSE_ATTN_FORCE_BUILD="TRUE"
export MAX_JOBS="8"
export NVCC_THREADS="1"

"$toolchain_root/bin/nvcc" --version
if ! "$environment_root/bin/python" -c 'import torch; import block_sparse_attn' >/dev/null 2>&1; then
    "$uv_binary" pip install \
        --python "$environment_root/bin/python" \
        --no-build-isolation \
        --no-deps \
        "$block_sparse_root"
fi

"$environment_root/bin/python" - <<'PY'
import torch

if torch.__version__ != "2.6.0+cu124":
    raise RuntimeError(f"unexpected torch version: {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the FlashVSR worker environment")
if torch.cuda.get_device_capability(0) != (8, 6):
    raise RuntimeError(
        f"unexpected GPU compute capability: {torch.cuda.get_device_capability(0)}"
    )
print(
    "FlashVSR base runtime ready:",
    torch.__version__,
    torch.cuda.get_device_name(0),
    torch.cuda.get_device_capability(0),
)
PY

"$environment_root/bin/python" "$repo_root/scripts/smoke_block_sparse.py"
PYTHONPATH="$source_root" "$environment_root/bin/python" - <<'PY'
from diffsynth import FlashVSRTinyPipeline, ModelManager

print("FlashVSR source import passed:", ModelManager.__name__, FlashVSRTinyPipeline.__name__)
PY
