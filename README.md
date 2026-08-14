# video_enhance

RTX 3090에서 실제 영상을 `FPS 보간 → Video Super Resolution` 순서로 처리하기 위한 fail-fast 파이프라인입니다.

현재는 기반 계약, 모델 선정, 격리 worker protocol과 RTX 3090 모델 runtime을 구현하는
단계입니다. 최종 산출물은 설정 검증 단계부터 `/mnt/d` 아래로 제한합니다.

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
`max(video 1-frame, AAC 1-frame)`을 넘으면 결과를 확정하지 않습니다.

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

상세 설계는 [DESIGN.md](DESIGN.md), 모델 근거는 [MODEL_SELECTION.md](MODEL_SELECTION.md)를
참고합니다.
