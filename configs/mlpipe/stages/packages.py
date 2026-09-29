import json
import shutil
from pathlib import Path
from ultralytics import YOLO


def run(ctx):
    cfg = ctx.cfg
    dst = Path(cfg.package.dir) / cfg.run_name
    dst.mkdir(parents=True, exist_ok=True)

    names = YOLO(str(ctx.weights)).names
    meta = {
        "run_name": cfg.run_name,
        "imgsz": cfg.imgsz,
        "precision": cfg.export.precision,
        "classes": [names[i] for i in sorted(names)],
        "metrics": ctx.metrics,
    }
    if ctx.tflite:
        shutil.copy(ctx.tflite, dst / "model.tflite")
    if ctx.onnx:
        shutil.copy(ctx.onnx, dst / "model.onnx")
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[package] -> {dst}")