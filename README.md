# video_enhance

RTX 3090에서 실제 영상을 `FPS 보간 → Video Super Resolution` 순서로 처리하기 위한 fail-fast 파이프라인입니다.

현재는 기반 계약과 테스트를 구현한 초기 단계입니다. 모델 선정과 체크포인트 설치 전이므로 아직 실제 영상 추론을 실행하지 않습니다.

## 테스트

외부 패키지 없이 현재 기반 테스트를 실행할 수 있습니다.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

상세 설계와 단계별 완료 조건은 [DESIGN.md](DESIGN.md)를 참고합니다.
