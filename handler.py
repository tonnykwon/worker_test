import os

import runpod
from probe_lib import ensure_model, run_all

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/pose_landmarker_lite.task")


def handler(event):
    model_path = ensure_model(MODEL_PATH)
    return run_all(model_path)


runpod.serverless.start({"handler": handler})
