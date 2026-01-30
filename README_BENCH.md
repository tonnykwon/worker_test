# Video Benchmark

이 스크립트는 **CPU vs GPU 처리 속도 차이**를 비교합니다.

## 준비

- 컨테이너에 `/data/sample.mp4` 제공 (bind mount)
- `VIDEO_PATH` 환경변수로 경로 지정 가능

## 실행

```
VIDEO_PATH=/data/sample.mp4 python benchmark_video.py
```

## Modal

```
modal run modal_benchmark.py::run_benchmark
```
