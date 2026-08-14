# VFI/VSR 처리 순서 A/B benchmark

기준일은 2026-08-14입니다. 동일한 Practical-RIFE v4.25, MMagic RealBasicVSR,
2배 FPS/2배 해상도, CRF 16, BT.709 decode/encode 조건에서 처리 순서만 비교했습니다.

## 결론

현재 두 대표 클립에서는 `VSR → VFI`가 temporal stability와 실행시간에서 일관되게
우세합니다. `VFI → VSR`은 077에서 Laplacian detail이 더 높지만 midpoint overshoot와
high-frequency temporal variation도 함께 증가했습니다.

- temporal stability 우선 후보: `VSR → VFI`
- 정지 texture/detail 우선 후보: `VFI → VSR`
- 운영 기본값은 아직 `VFI → VSR` 유지: reverse 경로의 receipt-resume 구현 후 변경
- 자동 승자 판정 금지: 아래 값은 GT 없는 proxy이며 수동 motion artifact 검수와 함께 사용

## 실측 결과

| 입력 | 순서 | wall time | curvature | overshoot | Laplacian temporal Δ | source PSNR | 출력 크기 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 077, 604×1080, 191 CFR frames | VFI→VSR | 357s | 0.007850 | 0.390997 | 0.028621 | 30.6435dB | 22,077,970B |
| 077, 604×1080, 191 CFR frames | VSR→VFI | 289s | 0.006810 | 0.248518 | 0.024737 | 30.5871dB | 17,256,086B |
| 059 motion crop, 392×720, 90 frames | VFI→VSR | 65s | 0.003692 | 0.163057 | 0.015233 | 33.3154dB | 2,294,424B |
| 059 motion crop, 392×720, 90 frames | VSR→VFI | 45s | 0.003579 | 0.108899 | 0.014459 | 33.2769dB | 2,012,353B |

reverse의 forward 대비 변화는 다음과 같습니다.

- 077: curvature -13.2%, overshoot -36.4%, Laplacian temporal Δ -13.6%, wall time -19.0%
- 059 motion: curvature -3.1%, overshoot -33.2%, Laplacian temporal Δ -5.1%, wall time -30.8%
- source PSNR 감소는 각각 0.056dB, 0.038dB로 작았습니다.
- spatial sharpness는 077에서 forward even/odd가 약 7.1%/2.9% 높았고, 059에서는 reverse
  even이 약 2.1% 높고 odd는 사실상 같았습니다.

## 지표 정의와 한계

- `source_roundtrip_psnr_db`: 2배 output의 even frame을 정확한 2×2 box로 축소해 원본 CFR
  frame과 비교합니다. restoration detail의 perceptual quality를 직접 나타내지는 않습니다.
- `midpoint_curvature_mae`: odd VFI frame과 두 인접 even frame의 선형 midpoint 차이입니다.
- `midpoint_overshoot_rate`: odd frame이 인접 even frame의 local intensity 범위를 벗어난 비율입니다.
- `laplacian_abs_mean`: 정지 공간 detail proxy입니다. ringing도 높은 값으로 계산될 수 있습니다.
- `laplacian_temporal_delta_mae`: 고주파 flicker proxy이며 flow compensation 전 값입니다.
- 모든 temporal metric은 scene-cut transition을 제외합니다.

## 추가 gate

1920×1072 source 2프레임을 `VSR → VFI`로 처리해 3840×2144/60fps 결과를 만들었습니다.
RTX 3090에서 OOM 없이 15초에 통과했지만, 이는 고해상도 장시간 품질 검증을 대신하지 않습니다.

원시 metric JSON과 좌우 비교 영상은 `/mnt/d/Lewd/트위터r_enhanced/_smoke/`에 있습니다.
좌측은 forward, 우측은 reverse입니다.
