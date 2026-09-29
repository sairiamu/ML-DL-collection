from pathlib import Path
from ultralytics import YOLO


def _find_tflite(weights: Path, variant: str) -> Path:
    saved = weights.with_name(weights.stem + "_saved_model")
    hits = sorted(saved.glob(f"*_{variant}.tflite"))
    if not hits:
        raise FileNotFoundError(f"no *_{variant}.tflite in {saved}")
    return hits[0]


def run(ctx):
    cfg, e = ctx.cfg, ctx.cfg.export
    assert ctx.weights, "no weights: run finetune or set `weights:`"

    if e.onnx:
        m = YOLO(str(ctx.weights))
        p = m.export(format="onnx", imgsz=cfg.imgsz, opset=e.opset,
                     simplify=e.simplify)
        ctx.onnx = Path(p)

    if e.tflite:
        m = YOLO(str(ctx.weights))  # fresh instance; export mutates model state
        kw = dict(format="tflite", imgsz=cfg.imgsz)
        variant = "float32"
        if e.precision == "fp16":
            kw["half"], variant = True, "float16"
        elif e.precision == "int8":
            kw.update(int8=True, data=cfg.data, fraction=e.calib_fraction)
            variant = "int8"
        m.export(**kw)
        ctx.tflite = _find_tflite(Path(ctx.weights), variant)

    ctx.metrics["onnx"] = str(ctx.onnx) if ctx.onnx else None
    ctx.metrics["tflite"] = str(ctx.tflite) if ctx.tflite else None