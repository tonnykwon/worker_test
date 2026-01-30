import json
import os

from probe_lib import ensure_model, run_all


def main():
    model_path = ensure_model(
        os.environ.get("MODEL_PATH", "/models/pose_landmarker_lite.task")
    )
    result = run_all(model_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
