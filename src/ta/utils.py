"""Seed, info git, deteksi device."""
import os
import random
import subprocess

import numpy as np

from ta.paths import ROOT


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def machine_info() -> dict:
    """Info mesin untuk metadata run: hostname, OS, Python, versi library, GPU. Penting kalau run dipindah antar laptop."""
    import platform
    from importlib import metadata
    info = {"hostname": platform.node(), "os": platform.platform(), "python": platform.python_version(), "libs": {}}
    for lib in ("torch", "transformers", "bitsandbytes", "xgboost", "scikit-learn", "numpy", "pandas", "librosa"):
        try:
            info["libs"][lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_vram_gb"] = round(p.total_memory / 2**30, 2)
    except ImportError:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True, timeout=20).stdout.strip()
        info["nvidia_driver"] = out.splitlines()[0] if out else ""
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""
    return {"commit": run("rev-parse", "HEAD") or "none", "dirty": bool(run("status", "--porcelain", "--", "src", "configs"))}
