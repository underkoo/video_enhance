# video_enhance

RTX 3090에서 실제 영상을 `FPS 보간 → Video Super Resolution` 순서로 처리하기 위한 fail-fast 파이프라인입니다.

현재는 기반 계약, 모델 선정, 격리 worker protocol, RTX 3090 모델 runtime 및 단일 영상
end-to-end 실행 경로까지 구현했습니다. 최종 산출물은 설정 검증 단계부터 `/mnt/d` 아래로
제한합니다.

30fps 재타이밍은 고정 FFmpeg의 실제 filter summary를 먼저 검증합니다. 현재 데이터셋의
6개 drift 영상에서 확인한 `(source → CFR, drop, duplicate)`는 각각
`986→984 (2,0)`, `1903→1901 (2,0)`, `658→656 (2,0)`, `4193→4196 (0,3)`,
`1151→1149 (3,1)`, `190→191 (0,1)`입니다.

현재 104개 입력은 모두 color tag가 없는 `yuv420p`입니다. 기본 설정은 이를 BT.709 limited로
명시해 full-range RGB24로 변환합니다. 077 영상의 191-frame stream을 독립 2회 실행한 결과
동일 SHA-256 `bc3d116c99578f0d5f26f8507010a1be51de2db525d9fb14ade65cc9ac77506c`를
확인했습니다.

RIFE 입력은 scene을 넘지 않으며 최대 64 source frame, 경계 1-frame overlap으로 조립합니다.
077 영상의 CFR 191프레임은 `64/64/64/2` frame NPY 네 개로 검증했고, 중간 chunk는 WSL의
`.runtime` 아래에만 생성합니다.

077 실제 영상 결합 smoke에서는 RIFE worker 네 개가 29.66초 내 완료·병합됐고, 382-frame
보간 timeline을 FlashVSR 21-frame 입력 24개로 gap/overlap 없이 재조립했습니다.

기존 실해상도 FlashVSR 21-frame 결과는 `/mnt/d`에 1208×2160, 60fps, H.264 CRF 16으로
atomic encode했습니다. 출력은 21프레임/0.35초이고 BT.709 limited의 range/matrix/transfer/
primaries가 모두 ffprobe 검증을 통과했습니다.

동일 smoke의 AAC copy 경로도 검증했습니다. video는 0.350000초, 44.1kHz AAC는 packet
경계상 0.371519초이며 차이 21.519ms는 AAC 1-frame(23.220ms) 이내입니다. validator는
remux된 AAC duration이 원본 AAC duration과 1 AAC-frame 이상 다르면 결과를 확정하지
않습니다. 원본에 존재하는 video/audio duration 차이를 오디오 재인코딩으로 숨기지 않습니다.

RealBasicVSR는 MMagic v1.2.0의 inference architecture와 공식 EMA checkpoint를 최소
PyTorch worker로 격리했습니다. RTX 3090 실측 결과 604×1080×3 입력은 1.76초/peak
reserved 7.70GB, 1920×1072×2 입력은 3.04초/23.45GB였습니다. 후자는 여유가 작으므로
해상도별 padded frame-pixel 계약을 넘는 요청을 사전 거부합니다.

077 영상의 실제 end-to-end smoke는 `191 CFR frame → 382 RIFE frame → 95 RealBasicVSR
chunk`를 거쳐 1208×2160/60fps/382-frame MP4를 생성했습니다. 결과는 22,077,970 bytes,
SHA-256 `a4449acb124a5b56565b1ced6a80040734f41accd9cea4c67cd49b311842f893`이며
BT.709 limited와 원본 44.1kHz AAC duration 보존을 검증했습니다.

RealBasicVSR worker는 JSON Lines persistent session에서 checkpoint를 한 번만 적재합니다.
동일 077 end-to-end 실행은 389초에 완료됐고, 기존 청크별 프로세스 결과와 MP4 전체가
byte-identical했습니다. 청크별 terminal response와 실제 NPY digest/shape 검증은 persistent
경로에서도 생략하지 않습니다.

중단 재개는 NPY 존재 여부만 신뢰하지 않습니다. 각 artifact의 producer fingerprint,
SHA-256, shape, dtype을 `*.receipt.json`에 atomic 기록합니다. 재개 시 입력 파일 SHA,
resolved config SHA, CFR drop/dup, scene cut과 chunk plan을 `run-plan.json`과 먼저 비교한 뒤
receipt가 완전히 검증된 청크만 재사용합니다. 077 영상을 VSR 3청크 직후 중단하고 재개한
실제 smoke 결과도 정상 실행 MP4와 byte-identical했습니다.

## 테스트

Python 3.12 제어 환경을 만들고 테스트합니다.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/mypy
```

## RIFE 격리 환경

고정된 Python 3.11.13/CUDA 12.4 worker 환경, upstream commit, checkpoint를 설치하고
동일 입력 2회 byte 결정성까지 검사합니다. 모든 대용량 runtime 파일은 `.runtime/` 아래에만
생성되며 Git에는 포함되지 않습니다.

```bash
scripts/bootstrap_rife.sh
```

FFmpeg/ffprobe는 변경 가능한 `latest` URL을 사용하지 않고 고정 release asset과 SHA-256으로
`.runtime/tools/`에 설치합니다.

```bash
scripts/bootstrap_ffmpeg.sh
```

FlashVSR v1.1의 고정 Python/CUDA 환경, 4개 checkpoint, source-built Block Sparse Attention을
설치합니다. Bootstrap은 각 checkpoint의 크기와 SHA-256을 검증한 뒤 RTX 3090에서 BF16
kernel 결과를 PyTorch SDPA reference와 비교합니다.

```bash
scripts/bootstrap_flashvsr.sh
```

실제 diffusion checkpoint까지 포함한 최소 21-frame 추론 smoke는 별도로 실행합니다.

```bash
PYTHONPATH=src .runtime/envs/flashvsr-v1.1/bin/python scripts/smoke_flashvsr.py
```

실제 MP4 해상도와 21-frame chunk의 VRAM 적합성은 다음처럼 별도 검증합니다.

```bash
PYTHONPATH=src .runtime/envs/flashvsr-v1.1/bin/python \
  scripts/smoke_flashvsr_real.py --input /absolute/path/to/input.mp4 --output-scale 2
```

RealBasicVSR 고정 환경과 checkpoint를 설치하고 결정론적 smoke를 실행합니다.

```bash
scripts/bootstrap_realbasicvsr.sh
```

검증된 운영 baseline으로 단일 영상을 처리합니다. 최종 출력은 반드시 `/mnt/d` 아래의 새
경로여야 하며, 실패 시 중간 NPY는 `.runtime/jobs/`에 보존됩니다.
성공 시 출력과 같은 basename의 `.provenance.json`에 두 모델의 upstream commit과 checkpoint
SHA-256을 atomic 기록합니다. `.run.json`에는 입력/config SHA, CFR/scene/chunk plan,
최종 MP4 SHA·크기·해당 실행 wall time을 기록합니다.

```bash
PYTHONPATH=src .venv/bin/python scripts/enhance_video.py \
  --input '/mnt/d/Lewd/트위터r/input.mp4' \
  --output '/mnt/d/Lewd/트위터r_enhanced/input_enhanced.mp4' \
  --config deterministic
```

실패 로그에 출력된 작업 디렉터리에서 재개할 때는 입력·출력·config를 바꾸지 않고 다음처럼
실행합니다.

```bash
PYTHONPATH=src .venv/bin/python scripts/enhance_video.py \
  --input '/mnt/d/Lewd/트위터r/input.mp4' \
  --output '/mnt/d/Lewd/트위터r_enhanced/input_enhanced.mp4' \
  --config deterministic \
  --resume-work '/workspace/01_Codes/github/video_enhance/.runtime/jobs/enhance-xxxxxxxx'
```

재개 동작이나 제한된 청크 실행을 검증하려면 `--checkpoint-after-vsr-chunks N`으로 N개의 신규
VSR receipt가 확정된 직후 의도적으로 중단할 수 있습니다.

상세 설계는 [DESIGN.md](DESIGN.md), 모델 근거는 [MODEL_SELECTION.md](MODEL_SELECTION.md)를
참고합니다.
