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
12. resume는 artifact 존재만으로 허용하지 않습니다. immutable run plan과 producer/content
    receipt가 모두 일치한 청크만 재사용합니다.

## 4. 처리 단계

### 4.1 Probe 및 정규화

- codec, width, height, rational FPS, time base, frame count, duration, color metadata, audio stream을 수집합니다.
- 컨테이너 FPS와 실제 PTS cadence가 일치하는지 검사합니다.
- 재타이밍은 `setpts=PTS-STARTPTS` 뒤 정확한 `30/1`, `round=near`, `eof_action=pass`로 고정합니다.
- 동일 FFmpeg filter를 null sink로 먼저 실행해 실제 input/output/drop/dup count를 확정한 뒤에만
  RGB decode를 허용합니다.
- CFR 정규화의 frame duplication/drop 내역을 manifest에 기록합니다.
- 현재 104개 입력은 모두 tag 없는 `yuv420p`입니다. 기본 decode 가정은 BT.709 limited로
  설정에 노출하고 FFmpeg filter에 강제하며, 가정 적용 여부를 manifest에 기록합니다.
- RGB24는 전체 영상을 적재하지 않고 frame callback으로 스트리밍하며 short/extra byte를 즉시
  실패시킵니다. 최종 인코딩은 BT.709 tag를 명시합니다.

### 4.2 Scene-cut-aware VFI

- 각 인접 프레임 쌍의 scene cut 여부를 먼저 결정합니다.
- 고정 FFmpeg `scdet`에서 모든 프레임 score를 추출한 뒤 Python에서 명시적 임계값을 적용합니다.
- 초기 임계값 27.0은 001 영상 1,515프레임 전수 스캔(max 12.87)에서 fast-motion 오검출이 없음을
  확인했으며, 전체 대표 클립 검증 전까지 후보값으로 취급합니다.
- 일반 구간만 VFI backend에 전달합니다.
- cut 구간은 보간하지 않고 이전 프레임을 유지합니다.
- 마지막 프레임 terminal hold를 추가해 오디오와 표시 시간을 보존합니다.
- 긴 scene은 최대 64 source-frame chunk로 나누고 경계 source frame 1장을 공유합니다. 앞 chunk의
  잘못된 terminal hold와 뒤 chunk의 중복 source frame을 ownership slice에서 제거합니다.
- 1-frame scene은 RIFE 호출이 불가능하므로 모델을 우회하고 해당 frame을 multiplier만큼 hold합니다.

### 4.3 Video SR

- frame-independent image SR이 아니라 시간축 정보를 사용하는 Video SR backend를 기본으로 합니다.
- RIFE worker output은 chunk별 ownership slice만 순차 방출하고, FlashVSR 입력은 scene별
  21-frame window와 5-frame left context로 다시 조립합니다.
- FlashVSR worker output에서도 left context와 terminal padding을 제거하고 각 chunk의 전역
  ownership만 encoder에 전달합니다.
- RealBasicVSR 운영 baseline은 SPyNet 32-pixel padding을 포함한
  `model_frames×padded_width×padded_height`를 RTX 3090 실측 상한 4,177,920 이하로
  제한합니다. 1920×1072는 2-frame pair, 604×1080은 최대 6-frame 문맥을 사용합니다.
- RealBasicVSR chunk는 scene을 넘지 않고 좌·우 context ownership을 제거합니다. 1-frame
  scene만 동일 frame을 terminal padding해 최소 2-frame 모델 입력을 만족합니다.
- FlashVSR 고해상도 spatial tile은 아직 seam 품질 검증 전이므로 자동 실행하지 않습니다.
- 생성형 세부 복원 강도는 기본값에서 보수적으로 제한합니다.

### 4.4 인코딩 및 오디오

- 검증용 lossless/intermediate와 배포용 encode를 분리할 수 있게 합니다.
- 기본 배포 encode는 libx264 slow/CRF 16, `yuv420p`, BT.709 limited입니다. container tag와
  x264 bitstream VUI를 모두 설정합니다.
- 오디오는 재생 속도를 변경하지 않고 원본 stream을 remux합니다.
- 비디오·오디오 duration 차이는 `max(출력 1 frame, AAC 1 frame)` 이내만 허용합니다.
- `.partial.mp4`의 codec, 크기, frame count, rational FPS, video duration, 네 color tag, audio
  존재 여부를 count-decode ffprobe로 검증한 뒤에만 atomic rename합니다.

## 5. 코드 구조

```text
src/rvfi_sr/
  artifacts.py       # atomic output 및 overwrite 방지
  geometry.py        # alignment pad/crop 계약
  timeline.py        # CFR, scene cut, frame-count 계약
  probe.py           # ffprobe JSON parser
  config.py          # Pydantic 설정 검증
  scene_cut.py       # strict FFmpeg scdet parser 및 transition index
  rife_chunks.py     # scene-safe RIFE overlap/output ownership
  temporal_chunks.py # scene 경계를 넘지 않는 FlashVSR chunk ownership
  chunk_io.py        # bounded-memory atomic NPY input assembly
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

- 완료: 로컬 인벤토리, 타임라인/geometry/artifact/probe 계약, 단위 테스트 106개
- 완료: 공식 소스 기반 모델 shortlist와 라이선스/runtime 격리 정책
- 완료: Pydantic/Hydra 제어 설정, backend protocol, RTX 3090/RIFE deterministic preflight
- 완료: FlashVSR v1.1 고정 환경, 4개 checkpoint digest, SM 8.6 sparse CUDA kernel 수치 smoke
- 완료: FlashVSR synthetic 2회 byte repeatability 및 604×1080 실제 영상 21-frame smoke
- 완료: 고정 FFmpeg `scdet` strict parser, scene-safe 21-frame/5-context FlashVSR chunk planner
- 완료: 30/1 CFR preflight 및 실제 6개 drift 영상의 strict drop/dup accounting
- 완료: 077 영상 191-frame BT.709 limited RGB stream 독립 2회 byte 결정성 검증
- 완료: 077 영상의 4개 overlapping RIFE NPY input을 bounded memory/atomic write로 실측 조립
- 완료: 077 영상 실제 RIFE 4-worker 결과를 ownership merge해 FlashVSR 입력 24개로 조립
- 완료: FlashVSR context/padding 제거와 전역 output ownership merge 계약
- 완료: 1208×2160/60fps/21-frame MP4 atomic encode, BT.709 VUI, AAC remux/duration 검증
- 완료: 공식 MMagic RealBasicVSR EMA checkpoint와 strict 320-tensor load 계약
- 완료: 604×1080×3 및 1920×1072×2 RealBasicVSR 2× 실해상도/VRAM/결정성 smoke
- 완료: 단일 영상 RIFE→RealBasicVSR→AAC remux end-to-end CLI와 077 영상 382-frame smoke
- 완료: RealBasicVSR checkpoint 1회 load persistent worker, 077 영상 389초 및 기존 MP4와
  byte-identical 검증
- 완료: 입력/config/run-plan과 청크별 NPY receipt 기반 resume, 의도적 3청크 중단 후 077
  byte-identical 완주
- 완료: 최종 `.run.json`의 입력/output SHA, CFR/scene/chunk plan, wall time 기록
- 완료: `VFI→VSR`/`VSR→VFI` research preset, 동일 timeline/color/audio 검증
- 완료: 077 전체 영상과 059 fast-motion crop의 GT 없는 temporal/spatial A/B
- 완료: 1920×1072→3840×2144 reverse 2-frame RTX 3090 OOM gate
- 진행: reverse receipt-resume와 전체 batch readiness
- 미실행: FlashVSR spatial tile 품질 검증, A/B benchmark, 전체 batch 처리
