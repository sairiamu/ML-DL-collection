import json
from pathlib import Path
from omegaconf import OmegaConf
from .context import Context
from .stages import STAGES


def load_config(path: str, overrides: list[str] | None = None):
    p = Path(path)
    cfg = OmegaConf.load(p)
    base_name = cfg.pop("defaults_from", None)
    if base_name:
        cfg = OmegaConf.merge(OmegaConf.load(p.parent / base_name), cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def run_pipeline(cfg):
    out = Path(cfg.out_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    ctx = Context(cfg=cfg, out_dir=out,
                  weights=Path(cfg.weights) if cfg.weights else None)
    for name in cfg.pipeline:
        print(f"\n=== stage: {name} ===")
        STAGES[name](ctx)
    (out / "report.json").write_text(json.dumps(ctx.metrics, indent=2))
    OmegaConf.save(cfg, out / "config.resolved.yaml")
    return ctx