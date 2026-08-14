"""RTX 3090에서 Block Sparse Attention CUDA kernel을 수치 검증합니다."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from block_sparse_attn import block_sparse_attn_func


def main() -> int:
    """BF16 dense-mask kernel을 PyTorch SDPA reference와 비교합니다."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_capability(0) != (8, 6):
        raise RuntimeError(
            f"unexpected GPU capability: {torch.cuda.get_device_capability(0)}"
        )

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    sequence_length = 256
    head_count = 4
    head_dimension = 64
    q = torch.randn(
        sequence_length,
        head_count,
        head_dimension,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cumulative_lengths = torch.tensor(
        [0, sequence_length], device=device, dtype=torch.int32
    )
    head_mask_type = torch.zeros(head_count, device=device, dtype=torch.int32)

    with torch.inference_mode():
        actual = block_sparse_attn_func(
            q,
            k,
            v,
            cumulative_lengths,
            cumulative_lengths,
            head_mask_type,
            None,
            None,
            sequence_length,
            sequence_length,
            0.0,
            deterministic=True,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=False,
        )
        expected = functional.scaled_dot_product_attention(
            q.permute(1, 0, 2).unsqueeze(0),
            k.permute(1, 0, 2).unsqueeze(0),
            v.permute(1, 0, 2).unsqueeze(0),
        ).squeeze(0).permute(1, 0, 2)

    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise RuntimeError(
            f"kernel output contract mismatch: shape={actual.shape}, dtype={actual.dtype}"
        )
    max_error = (actual.float() - expected.float()).abs().max().item()
    if max_error > 0.02:
        raise RuntimeError(f"Block Sparse Attention max error is too large: {max_error}")

    print(
        "Block Sparse Attention smoke passed:",
        f"gpu={torch.cuda.get_device_name(0)!r}",
        f"shape={tuple(actual.shape)}",
        f"dtype={actual.dtype}",
        f"max_abs_error={max_error:.8f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
