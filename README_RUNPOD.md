# RunPod Serverless 테스트

## 개요

GitHub Integration으로 `worker_test`를 그대로 올려 **GPU/EGL/MediaPipe delegate** 확인.

## 필수 파일

- `Dockerfile.runpod`
- `handler.py`
- `probe_lib.py`
- `requirements.txt`

## GitHub Integration 설정

RunPod 콘솔에서:

- Repository: `backend` 포함된 현재 repo
- Dockerfile Path: `worker_test/Dockerfile.runpod`
- Context: repo root
- Start Command: 기본 CMD 사용

## 환경변수

- `MODEL_PATH` (선택)
  - 기본값: `/models/pose_landmarker_lite.task`

## 실행 결과

`handler.py`가 JSON 결과를 반환합니다:

- `probe_env`
- `probe_mediapipe`

## 판정 기준

- `probe_env.eglinfo.ok == true`
- `probe_env.egl_py.ok == true`
- `probe_mediapipe.gpu.ok == true`

이 3개가 True면 GPU delegate 가능.
