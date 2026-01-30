FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ca-certificates curl git \
    libegl1 libgles2 libgl1 libgl1-mesa-dri \
    mesa-utils mesa-utils-extra \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Download a small MediaPipe Tasks model at build time (can be overridden by MODEL_PATH)
RUN mkdir -p /models && \
    curl -L -o /models/pose_landmarker_lite.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

COPY probe_env.py /app/probe_env.py
COPY probe_mediapipe.py /app/probe_mediapipe.py
COPY run_all.py /app/run_all.py

ENV MODEL_PATH=/models/pose_landmarker_lite.task

CMD ["python3", "/app/run_all.py"]
