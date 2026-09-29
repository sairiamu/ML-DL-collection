from pathlib import Path
from ultralytics import YOLO


def run(ctx):
    cfg = ctx.cfg
    f = cfg.finetune

    if f.strategy == "lora":
        # YOLO is conv-heavy; LoRA gains are unproven here. Slot kept so the
        # interface is stable. Implement with peft for RT-DETR/ViT backbones.
        raise NotImplementedError("lora: use strategy=freeze for YOLO")

    freeze = f.freeze_layers if f.strategy == "freeze" else None

    model = YOLO(f.base_weights)
    model.train(
        data=cfg.data,
        epochs=f.epochs,
        imgsz=cfg.imgsz,
        batch=f.batch,
        patience=f.patience,
        freeze=freeze,
        device=cfg.device,
        project=str(ctx.out_dir),
        name="train",
        exist_ok=True,
        **dict(f.extra),
    )
    ctx.weights = Path(model.trainer.best)
    ctx.metrics["train_best"] = str(ctx.weights)