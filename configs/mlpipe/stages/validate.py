import time
import numpy as np
from ultralytics import YOLO

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter


def _map(path, cfg):
    r = YOLO(str(path), task="detect").val(
        data=cfg.data, imgsz=cfg.imgsz, batch=1, device="cpu", verbose=False)
    return {"map50": float(r.box.map50), "map50_95": float(r.box.map)}


def _latency(path, runs, threads):
    it = Interpreter(model_path=str(path), num_threads=threads)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    x = np.random.rand(*inp["shape"]).astype(inp["dtype"]) \
        if inp["dtype"] == np.float32 else \
        np.random.randint(0, 255, inp["shape"]).astype(inp["dtype"])
    for _ in range(5):
        it.set_tensor(inp["index"], x); it.invoke()
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        it.set_tensor(inp["index"], x); it.invoke()
        t.append((time.perf_counter() - s) * 1000)
    out = it.get_output_details()[0]
    return {"ms_p50": float(np.percentile(t, 50)),
            "ms_p95": float(np.percentile(t, 95)),
            "input_shape": [int(v) for v in inp["shape"]],
            "input_dtype": np.dtype(inp["dtype"]).name,
            "output_shape": [int(v) for v in out["shape"]]}


def run(ctx):
    cfg, v = ctx.cfg, ctx.cfg.validate
    base = _map(ctx.weights, cfg)
    ctx.metrics["pt"] = base
    if ctx.tflite:
        tf = _map(ctx.tflite, cfg)
        tf.update(_latency(ctx.tflite, v.latency_runs, v.threads))
        tf["map50_95_drop"] = base["map50_95"] - tf["map50_95"]
        ctx.metrics["tflite"] = tf
        print(f"[validate] pt mAP50-95={base['map50_95']:.4f} | "
              f"tflite={tf['map50_95']:.4f} | p50={tf['ms_p50']:.1f}ms")