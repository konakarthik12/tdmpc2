try:
    import hydra # type: ignore
except ModuleNotFoundError:
    print("install hydra first: pip install hydra-core")
from hydra import compose, initialize # type: ignore
import sys
with initialize(job_name="test_app", config_path='../tdmpc2', version_base=None):
    cfg = compose(config_name="config", overrides=sys.argv[1:])

print(cfg.task)
