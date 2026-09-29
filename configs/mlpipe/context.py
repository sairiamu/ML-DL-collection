from dataclasses import dataclass, field
from pathlib import Path
from omegaconf import DictConfig


@dataclass
class Context:
    cfg: DictConfig
    out_dir: Path
    weights: Path | None = None
    onnx: Path | None = None
    tflite: Path | None = None
    metrics: dict = field(default_factory=dict)