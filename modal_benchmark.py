import json
import os
import time
import urllib.request

import modal

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libegl1",
        "libgles2",
        "libgl1",
        "libgl1-mesa-dri",
        "mesa-utils",
        "mesa-utils-extra",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
    )
    .pip_install(
        "mediapipe==0.10.14",
        "numpy==1.26.4",
        "PyOpenGL==3.1.7",
        "PyOpenGL-accelerate==3.1.7",
        "opencv-python-headless==4.10.0.84",
    )
)

app = modal.App("mediapipe-benchmark")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/"
    "float16/latest/gesture_recognizer.task"
)


def _ensure_model(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def _run_video(model_path: str, video_path: str, delegate):
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base = python.BaseOptions(model_asset_path=model_path, delegate=delegate)
    opts = vision.GestureRecognizerOptions(base_options=base)
    recognizer = vision.GestureRecognizer.create_from_options(opts)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    t0 = time.time()
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        _ = recognizer.recognize(mp_img)

    cap.release()
    dt = time.time() - t0
    fps = frames / dt if dt > 0 else 0.0
    return {"frames": frames, "seconds": dt, "fps": fps}


@app.function(image=IMAGE, gpu="T4", timeout=1200)
def run_benchmark():
    model_path = _ensure_model("/models/gesture_recognizer.task")
    video_path = os.environ.get("VIDEO_PATH", "/data/sample.mp4")

    from mediapipe.tasks import python

    out = {"cpu": None, "gpu": None}
    try:
        out["cpu"] = _run_video(model_path, video_path, python.BaseOptions.Delegate.CPU)
    except Exception as e:
        out["cpu"] = {"error": str(e)}

    try:
        out["gpu"] = _run_video(model_path, video_path, python.BaseOptions.Delegate.GPU)
    except Exception as e:
        out["gpu"] = {"error": str(e)}

    result = json.dumps(out, indent=2)
    print(result)
    return result
    return result
