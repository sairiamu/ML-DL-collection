import sys
from mlpipe.runner import load_config, run_pipeline

if __name__ == "__main__":
    cfg = load_config(sys.argv[1], sys.argv[2:])
    run_pipeline(cfg)