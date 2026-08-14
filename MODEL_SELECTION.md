# 모델 선정 기록

기준일은 2026-08-14입니다. 논문 순위만으로 고르지 않고 공식 코드·체크포인트,
라이선스, RTX 3090 24GB 실행 가능성, temporal consistency를 함께 평가했습니다.

## 결론

- 기본 VFI: [Practical-RIFE v4.25](https://github.com/hzwer/Practical-RIFE)
- 연구 품질 VFI: [BiM-VFI](https://github.com/KAIST-VICLab/BiM-VFI), 명시적 제한 라이선스 opt-in 필요
- 기본 VSR 후보: [FlashVSR v1.1](https://github.com/OpenImagingLab/FlashVSR), RTX 3090 CUDA kernel preflight 통과 조건
- 결정론적 VSR fallback: [MMagic RealBasicVSR](https://github.com/open-mmlab/mmagic)
- 제외: [SeedVR2](https://github.com/ByteDance-Seed/SeedVR)는 공식 권장 VRAM이 24GB 단일 GPU 범위를 크게 넘습니다.

`default.yaml`은 RIFE → FlashVSR를 요청하지만 FlashVSR의 block-sparse CUDA extension을
RTX 3090에서 compile/smoke test하기 전에는 실제 추론을 허용하지 않습니다. preflight 실패 시
자동 fallback하지 않고 실행을 실패시킵니다. 사용자가 `deterministic.yaml`을 명시했을 때만
RealBasicVSR로 전환합니다.

현재 Block-Sparse-Attention upstream commit `49d6c39e4dc0303442cda3bb758b3925d4399c49`의
device dispatch는 SM 8.x에서 SM 8.6/8.9를 명시적으로 구분합니다. 따라서 RTX 3090을 코드상
배제하지는 않지만, FlashVSR 공식 문서는 A100/A800 이외 GPU의 호환성을 보장하지 않으므로
이 사실만으로 실행 가능 판정을 내리지 않습니다.

## 비교표

| 모델 | 용도 | 공식 환경/특성 | 라이선스 정책 | 결정 |
|---|---|---|---|---|
| Practical-RIFE v4.25 | VFI | 빠른 arbitrary-timestep 계열, Python ≤3.11 권장 | MIT | 운영 기본선 |
| [GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI) R-P | VFI | 연속 motion representation, 오래된 PyTorch/CuPy 격리 필요 | 비상업 조건 | 선택적 benchmark |
| BiM-VFI | VFI | CVPR 2025, 비균일 motion의 perceptual interpolation 지향 | 연구·교육 조건 | 품질 benchmark |
| FlashVSR v1.1 | real-world VSR | one-step streaming diffusion, native 4×, 별도 sparse CUDA kernel | Apache-2.0 | 조건부 기본 후보 |
| RealBasicVSR | real-world VSR | 결정론적 recurrent baseline, MMagic adapter 사용 | Apache-2.0 | 명시적 fallback |
| SeedVR2-3B | restoration/VSR | 공식 예시는 H100 80GB급을 전제로 함 | Apache-2.0 | RTX 3090 대상 제외 |

## 해상도 정책

검토한 실전 real-world VSR backend는 native 4×가 중심입니다. 원본 집합에는 이미 720p와
1080p 영상이 많고 D: 여유 공간이 약 223GB이므로 4× 영상을 전부 저장하는 것은 부적절합니다.
기본 목표는 2×이며, native 4× 출력을 디스크에 저장하지 않고 메모리에서 2×로 downsample합니다.
이 경로는 `post_downsample: true`가 없으면 validation 단계에서 거부됩니다. native 4×와 4→2×의
detail/aliasing/temporal flicker는 smoke set에서 별도로 비교합니다.

## 런타임 분리

제어 계층은 Python 3.12 + Hydra + Pydantic v2로 고정합니다. 모델별 공식 환경은 서로 충돌하므로
각 backend를 별도 Python/CUDA 환경과 별도 프로세스로 실행합니다. 오케스트레이터는 모델을 직접
import하지 않고 버전이 있는 JSON request/response만 교환합니다.

각 실행 manifest에는 다음을 반드시 기록합니다.

- upstream HTTPS URL과 정확한 40자리 Git commit
- checkpoint HTTPS URL과 SHA-256
- license 이름 및 제한 사용 opt-in 여부
- native scale, 최종 scale, post-downsample 여부
- 실제 FP16/tile/chunk 값과 peak VRAM

공식 artifact의 불변 식별자는 코드 registry에 고정했습니다. RIFE v4.25 zip은
22,919,050 bytes, SHA-256 `e63d481b7ae5d4a4e6ad7ac5b410ff78f3bf7be3b51b2e38ca8152747abde5b4`입니다.
FlashVSR v1.1의 네 weight는 Hugging Face revision
`ad1aceeac60dbd288e51acea9096b821a8703bee`에 고정했고 총 6,948,393,553 bytes입니다.

## 다음 검증 gate

1. FlashVSR sparse CUDA extension을 SM 8.6 대상으로 compile-only 검사합니다.
2. 16프레임 저해상도 synthetic clip으로 입출력 shape, NaN/Inf, 반복 결정성을 검사합니다.
3. 실제 영상 3개 짧은 crop에서 RIFE/FlashVSR smoke test를 수행합니다.
4. BiM-VFI는 제한 라이선스 opt-in을 받은 연구 preset에서만 RIFE와 A/B합니다.
5. VFI→VSR와 VSR→VFI를 temporal metric 및 수동 artifact로 비교한 뒤 순서를 확정합니다.
