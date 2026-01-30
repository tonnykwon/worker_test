# MediaPipe GPU Delegate Smoke Test

이 폴더는 **MediaPipe Tasks GPU delegate**가 실제로 가능한지 빠르게 확인하는 스모크 테스트입니다.

## 구성

- `probe_env.py`: GPU 디바이스, EGL 초기화 가능 여부 검사
- `probe_mediapipe.py`: MediaPipe Tasks GPU delegate로 1장 inference 가능 여부 검사
- `run_all.py`: 위 2개를 순차 실행하고 JSON 출력
- `Dockerfile`: 동일 컨테이너를 여러 플랫폼에 그대로 배포
- `modal_app.py`: Modal 실행용 래퍼

## 로컬 테스트 (Docker)

```bash
# 빌드
Dockerfile은 worker_test 기준

# 실행 (GPU)
docker run --rm --gpus all \
  -e MODEL_PATH=/models/gesture_recognizer.task \
  mediapipe-probe:latest
```

## Modal 실행

```bash
modal run modal_app.py::run_probe
```

## 판정 기준

- `probe_env.py`:
  - `nvidia_smi.ok == true`
  - `egl_py.ok == true`
- `probe_mediapipe.py`:
  - `gpu.ok == true`

=> **MediaPipe GPU 가능**

## 참고

- 모델은 `gesture_recognizer.task`를 사용합니다.
- 모델 경로는 `MODEL_PATH`로 지정 가능.
