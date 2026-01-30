import glob
import json
import os
import subprocess
import time
import urllib.request
from typing import Any

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/"
    "float16/latest/pose_landmarker_lite.task"
)


def ensure_model(path: str, url: str = MODEL_URL) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def probe_env() -> dict[str, Any]:
    def sh(cmd):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            return {"ok": True, "out": out.strip()}
        except Exception as e:
            return {"ok": False, "out": str(e)}

    def egl_py_probe():
        try:
            import ctypes

            from OpenGL import EGL

            dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
            major = ctypes.c_int()
            minor = ctypes.c_int()
            ok = EGL.eglInitialize(dpy, ctypes.byref(major), ctypes.byref(minor))
            return {"ok": bool(ok), "version": f"{major.value}.{minor.value}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return {
        "dev_nvidia": glob.glob("/dev/nvidia*"),
        "dev_dri": glob.glob("/dev/dri/*"),
        "nvidia_smi": sh(["bash", "-lc", "nvidia-smi -L"]),
        "eglinfo": sh(["bash", "-lc", "eglinfo | head -n 50"]),
        "egl_py": egl_py_probe(),
        "env": {
            k: os.environ.get(k)
            for k in [
                "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES",
                "CUDA_VISIBLE_DEVICES",
                "XDG_RUNTIME_DIR",
            ]
            if k in os.environ
        },
    }


def probe_mediapipe(model_path: str) -> dict[str, Any]:
    def run(delegate):
        base = python.BaseOptions(model_asset_path=model_path, delegate=delegate)
        opts = vision.PoseLandmarkerOptions(base_options=base)
        landmarker = vision.PoseLandmarker.create_from_options(opts)

        img = np.zeros((256, 256, 3), dtype=np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        t0 = time.time()
        res = landmarker.detect(mp_img)
        dt = (time.time() - t0) * 1000
        return {"ok": True, "ms": dt, "result_type": str(type(res))}

    out = {"cpu": None, "gpu": None}

    try:
        out["cpu"] = run(python.BaseOptions.Delegate.CPU)
    except Exception as e:
        out["cpu"] = {"ok": False, "error": str(e)}

    try:
        out["gpu"] = run(python.BaseOptions.Delegate.GPU)
    except Exception as e:
        out["gpu"] = {"ok": False, "error": str(e)}

    return out


def run_all(model_path: str) -> dict[str, Any]:
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
    return {
        "probe_env": probe_env(),
        "probe_mediapipe": probe_mediapipe(model_path),
    }
    }
