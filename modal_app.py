import glob
import json
import os
import subprocess
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
    )
)

app = modal.App("mediapipe-probe")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/"
    "float16/latest/pose_landmarker_lite.task"
)


def _ensure_model(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def _probe_env():
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
            ]
            if k in os.environ
        },
    }


def _probe_mediapipe(model_path: str):
    import time

    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

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


@app.function(image=IMAGE, gpu="T4", timeout=600)
def run_probe():
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
    model_path = _ensure_model("/models/pose_landmarker_lite.task")
    result = {
        "probe_env": _probe_env(),
        "probe_mediapipe": _probe_mediapipe(model_path),
    }
    out = json.dumps(result, indent=2)
    print(out)
    return out
