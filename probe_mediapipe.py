import json
import json
import os
import sys

from probe_lib import ensure_model, probe_mediapipe

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.getenv(
    "MODEL_PATH", "/models/pose_landmarker_lite.task"
)
    return {"ok": True, "ms": dt, "result_type": str(type(res))}


def main():
    model_path = ensure_model(MODEL)
    print(json.dumps(probe_mediapipe(model_path), indent=2))


if __name__ == "__main__":
    main()
