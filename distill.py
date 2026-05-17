import warnings

import hydra
from src.distillation.runner import run_distillation

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(
    version_base=None, config_path="src/configs", config_name="distill_soft_tokens"
)
def main(config):
    run_distillation(config)


if __name__ == "__main__":
    main()
