import json
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL = os.environ.get("MODEL_PATH", "/models/pose_landmarker_lite.task")
VIDEO = os.environ.get("VIDEO_PATH", "/data/sample.mp4")


def run_video(delegate):
    base = python.BaseOptions(model_asset_path=MODEL, delegate=delegate)
    opts = vision.GestureRecognizerOptions(base_options=base)
    recognizer = vision.GestureRecognizer.create_from_options(opts)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO}")

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


def main():
    out = {"cpu": None, "gpu": None}
    try:
        out["cpu"] = run_video(python.BaseOptions.Delegate.CPU)
    except Exception as e:
        out["cpu"] = {"error": str(e)}

    try:
        out["gpu"] = run_video(python.BaseOptions.Delegate.GPU)
    except Exception as e:
        out["gpu"] = {"error": str(e)}

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
