"""운영 후보 모델의 공식 source와 checkpoint digest를 고정합니다."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from rvfi_sr.provenance import CheckpointArtifact, ModelProvenance

_FLASHVSR_WEIGHT_REVISION = "ad1aceeac60dbd288e51acea9096b821a8703bee"
_FLASHVSR_WEIGHT_ROOT = (
    "https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/"
    f"{_FLASHVSR_WEIGHT_REVISION}"
)


MODEL_PROVENANCE: Mapping[str, ModelProvenance] = MappingProxyType(
    {
        "practical-rife-v4.25": ModelProvenance(
            backend_id="practical-rife-v4.25",
            upstream_url="https://github.com/hzwer/Practical-RIFE",
            upstream_commit="17d8c7a1005b37f4c97bfee04e316aaec7fdc536",
            license_name="MIT",
            checkpoints=(
                CheckpointArtifact(
                    filename="RIFEv4.25.zip",
                    url=(
                        "https://drive.usercontent.google.com/download?"
                        "id=1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg&export=download&confirm=t"
                    ),
                    sha256="e63d481b7ae5d4a4e6ad7ac5b410ff78f3bf7be3b51b2e38ca8152747abde5b4",
                    size_bytes=22_919_050,
                ),
            ),
        ),
        "flashvsr-v1.1": ModelProvenance(
            backend_id="flashvsr-v1.1",
            upstream_url="https://github.com/OpenImagingLab/FlashVSR",
            upstream_commit="b527c6f285fb30df530f5febc8b45764a789c961",
            license_name="Apache-2.0",
            checkpoints=(
                CheckpointArtifact(
                    filename="diffusion_pytorch_model_streaming_dmd.safetensors",
                    url=f"{_FLASHVSR_WEIGHT_ROOT}/diffusion_pytorch_model_streaming_dmd.safetensors",
                    sha256="bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be",
                    size_bytes=5_676_070_392,
                ),
                CheckpointArtifact(
                    filename="LQ_proj_in.ckpt",
                    url=f"{_FLASHVSR_WEIGHT_ROOT}/LQ_proj_in.ckpt",
                    sha256="d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2",
                    size_bytes=575_694_948,
                ),
                CheckpointArtifact(
                    filename="TCDecoder.ckpt",
                    url=f"{_FLASHVSR_WEIGHT_ROOT}/TCDecoder.ckpt",
                    sha256="e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161",
                    size_bytes=189_018_333,
                ),
                CheckpointArtifact(
                    filename="Wan2.1_VAE.pth",
                    url=f"{_FLASHVSR_WEIGHT_ROOT}/Wan2.1_VAE.pth",
                    sha256="38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
                    size_bytes=507_609_880,
                ),
            ),
        ),
        "mmagic-realbasicvsr": ModelProvenance(
            backend_id="mmagic-realbasicvsr",
            upstream_url="https://github.com/open-mmlab/mmagic",
            upstream_commit="c749dcc7172d198ac2a27c3e5a4d2181640f0fd5",
            license_name="Apache-2.0",
            checkpoints=(
                CheckpointArtifact(
                    filename="RealBasicVSR.pth",
                    url=(
                        "https://download.openmmlab.com/mmediting/restorers/"
                        "real_basicvsr/"
                        "realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_"
                        "20211104-52f77c2c.pth"
                    ),
                    sha256=(
                        "52f77c2c835aaa3fe675b3959b2f85010a6c6f63f77f7e279394646e55a4e376"
                    ),
                    size_bytes=148_239_017,
                ),
            ),
        ),
    }
)
