# video_enhance

RTX 3090에서 실제 영상을 `FPS 보간 → Video Super Resolution` 순서로 처리하기 위한 fail-fast 파이프라인입니다.

현재는 기반 계약, 모델 선정, 격리 worker protocol을 구현하는 단계입니다. 체크포인트 설치와
RTX 3090 preflight 전이므로 아직 실제 영상 추론을 실행하지 않습니다.

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

상세 설계는 [DESIGN.md](DESIGN.md), 모델 근거는 [MODEL_SELECTION.md](MODEL_SELECTION.md)를
참고합니다.
