# Real-world Video VFI → VSR 기술 설계

## 1. 목표

`D:\Lewd\트위터r`의 104개 실제 영상을 대상으로 다음 순서의 재현 가능한 오프라인 파이프라인을 구현합니다.

```text
probe → CFR normalization → scene-cut detection
→ frame interpolation → temporally consistent video SR
→ audio remux → artifact validation
```

기본 목표는 30fps 계열 입력을 60fps로 보간하고 공간 해상도를 2배로 확대하는 것입니다. perceptual detail보다 temporal stability와 원본 보존을 우선합니다.

## 2. 입력 및 실행 환경

- 파일: MP4 104개, 총 2.294GiB
- 총 재생시간: 약 12,052초(3시간 21분)
- 비디오: H.264 High Profile 104개
- FPS 표기: 30.0fps 98개, 약 29.9~30.09fps 6개
- 해상도: 28종, 392×720부터 1920×1072까지 혼재
- 오디오: AAC-LC 71개, 오디오 없음 33개
- GPU: RTX 3090 24GB, compute capability 8.6
- 런타임: WSL2 Ubuntu 24.04, Python 3.12.3
- 현재 Python 환경: PyTorch/OpenCV/Hydra/Pydantic 미설치
- WSL 내부 잔여 공간: 약 435GB
- D: 잔여 공간: 약 436GB

## 3. 강제 불변식

1. 입력 파일을 직접 수정하거나 덮어쓰지 않습니다.
2. 완성 파일은 `*.partial.mp4` 검증 후 atomic rename으로 확정합니다.
3. VFR 또는 비정규 FPS 입력은 추론 전에 명시적인 CFR로 정규화합니다.
4. 장면 전환을 가로질러 VFI 모델을 호출하지 않습니다.
5. 장면 전환의 중간 timestamp는 `hold_previous`로 채웁니다.
6. `N`개 입력을 `m`배 FPS로 만들 때 최종 프레임 수는 `N×m`입니다. `m(N-1)+1`개의 VFI 결과 뒤에 마지막 프레임을 `m-1`개 유지해 오디오와 표시 시간을 보존합니다.
7. 모델 alignment padding은 오른쪽·아래에만 적용하고, SR 출력은 정확히 `W×scale`, `H×scale`로 crop합니다.
8. OOM, NaN/Inf, 체크포인트 불일치, 프레임 누락, 색상 메타데이터 불일치는 경고로 무마하지 않고 실패시킵니다.
9. 자동 tile 축소나 fallback은 manifest에 실제 적용값을 기록합니다. 기록 없는 silent fallback은 금지합니다.
10. 전체 104개 처리는 대표 클립 smoke test와 예상 시간·출력 용량 보고가 통과하기 전 실행하지 않습니다.
11. 최종 영상 경로는 `/mnt/d` 아래만 허용하고, 모델·checkpoint·중간 smoke artifact는 WSL
    내부 `.runtime/`에 둡니다.

## 4. 처리 단계

### 4.1 Probe 및 정규화

- codec, width, height, rational FPS, time base, frame count, duration, color metadata, audio stream을 수집합니다.
- 컨테이너 FPS와 실제 PTS cadence가 일치하는지 검사합니다.
- CFR 정규화의 frame duplication/drop 내역을 manifest에 기록합니다.
- 디코딩 색공간을 명시적으로 고정하고 최종 인코딩에서 원본 색상 메타데이터를 복원합니다.

### 4.2 Scene-cut-aware VFI

- 각 인접 프레임 쌍의 scene cut 여부를 먼저 결정합니다.
- 일반 구간만 VFI backend에 전달합니다.
- cut 구간은 보간하지 않고 이전 프레임을 유지합니다.
- 마지막 프레임 terminal hold를 추가해 오디오와 표시 시간을 보존합니다.

### 4.3 Video SR

- frame-independent image SR이 아니라 시간축 정보를 사용하는 Video SR backend를 기본으로 합니다.
- 24GB VRAM을 넘는 입력은 spatial tile과 temporal chunk를 함께 사용합니다.
- tile overlap/crop은 출력 좌표계에서 자동 검증합니다.
- 생성형 세부 복원 강도는 기본값에서 보수적으로 제한합니다.

### 4.4 인코딩 및 오디오

- 검증용 lossless/intermediate와 배포용 encode를 분리할 수 있게 합니다.
- 오디오는 재생 속도를 변경하지 않고 원본 stream을 remux합니다.
- 비디오·오디오 duration 차이는 출력 프레임 1개 이내만 허용합니다.

## 5. 코드 구조

```text
src/rvfi_sr/
  artifacts.py       # atomic output 및 overwrite 방지
  geometry.py        # alignment pad/crop 계약
  timeline.py        # CFR, scene cut, frame-count 계약
  probe.py           # ffprobe JSON parser
  config.py          # Pydantic 설정 검증
  scene_cut.py       # cut detector 및 transition plan
  pipeline.py        # 단계 orchestration/resume
  backends/
    vfi_base.py
    vsr_base.py
    <selected adapters>
  validation/
    media.py
    frames.py
    manifest.py
configs/
  default.yaml
tests/
```

모델 코드는 adapter 경계 밖에 격리합니다. 체크포인트 SHA-256, upstream commit, license, 모델 파라미터를 run manifest에 기록합니다.

## 6. 모델 선정

공식 소스·라이선스·런타임 검토 결과와 선택 근거는 [MODEL_SELECTION.md](MODEL_SELECTION.md)에
고정합니다. 기본 VFI는 Practical-RIFE v4.25, 기본 VSR 후보는 RTX 3090 preflight를 전제로 한
FlashVSR v1.1입니다. 모든 모델은 Python 3.12 제어 계층과 분리된 프로세스에서 실행합니다.

### VFI

- fast motion, occlusion, thin structure, 반복 패턴 안정성
- scene-cut 입력을 외부에서 차단할 수 있는 API
- arbitrary resolution과 padding/crop 지원
- RTX 3090 FP16 추론 가능성
- 공식 코드·체크포인트·명확한 라이선스

### VSR

- real-world degradation 또는 blind video restoration 대응
- temporal consistency와 recurrent/propagation 안정성
- temporal chunk와 spatial tile 적용 가능성
- CUDA extension 및 Python 3.12 호환성
- 사용 조건이 명확한 라이선스

## 7. 검증 전략

### 단위 테스트

- rational FPS와 출력 프레임 수
- scene-cut hold 및 terminal hold
- alignment padding과 exact crop
- 입력/출력 alias 차단과 partial artifact 정책
- ffprobe payload와 설정 범위 검증

### 통합 테스트

- 합성 moving edge/occlusion/scene cut 클립
- 무음 및 AAC 오디오 입력
- 29.97/30fps, 다양한 종횡비, alignment 비정렬 해상도
- 강제 OOM/중단 후 resume
- 체크포인트 SHA-256 불일치

### 실제 영상 smoke set

- 빠른 카메라 이동
- 인물·피부·머리카락
- 가림과 재등장
- 화면 전환
- 압축 블록/링잉
- 작은 텍스트 또는 규칙적 패턴
- 세로 영상과 가로 영상

### 품질 비교

- `VFI→VSR` 기본안과 `VSR→VFI` 비교
- 시간축: flow-warp residual, temporal perceptual metric, flicker
- 공간축: no-reference IQA는 보조 지표로만 사용
- 수동 검수: ghosting, double edge, texture boiling, 얼굴/손 변형
- 성능: wall time, peak VRAM, output bytes, decode/encode 비중

## 8. 단계별 완료 조건

1. Foundation: 핵심 계약 테스트가 모두 통과합니다.
2. Model decision: 공식 소스·라이선스·체크포인트·호환성 표가 완성됩니다.
3. Dry run: 모델 없이 decode/scene plan/encode round-trip이 통과합니다.
4. VFI smoke: 실제 클립에서 cut crossing이 0건이고 프레임 수가 정확합니다.
5. VSR smoke: 목표 크기·tile seam·NaN/Inf 검증이 통과합니다.
6. A/B benchmark: 기본 순서와 프리셋을 근거와 함께 확정합니다.
7. Batch readiness: 예상 총 시간·용량을 보고한 뒤 전체 실행 여부를 결정합니다.

## 9. 현재 상태

- 완료: 로컬 인벤토리, 타임라인/geometry/artifact/probe 계약, 단위 테스트 56개
- 완료: 공식 소스 기반 모델 shortlist와 라이선스/runtime 격리 정책
- 완료: Pydantic/Hydra 제어 설정, backend protocol, RTX 3090/RIFE deterministic preflight
- 완료: FlashVSR v1.1 고정 환경, 4개 checkpoint digest, SM 8.6 sparse CUDA kernel 수치 smoke
- 완료: FlashVSR synthetic 2회 byte repeatability 및 604×1080 실제 영상 21-frame smoke
- 진행: 고정 FFmpeg/ffprobe 기반 CFR/scene-cut/chunking과 고해상도 VSR 전략
- 차단: 1280×720 이상은 현재 single-pass RTX 3090 실측 한도를 초과하므로 자동 실행 금지
- 미실행: spatial tile 품질 검증, 전체 batch 처리
