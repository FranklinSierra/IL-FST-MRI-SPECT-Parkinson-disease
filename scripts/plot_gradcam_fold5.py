#!/usr/bin/env python3
"""Grad-CAM (predicted class) for FST fold 5 on the official test set.

Thin wrapper around scripts/plot_gradcam_fst.py (defaults: primary fold 5).
"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "plot_gradcam_fst.py"), run_name="__main__")
