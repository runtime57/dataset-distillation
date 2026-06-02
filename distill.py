import warnings

import hydra
from src.distillation.runner import run_distillation

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="distill_soft_tokens_gm_10k_seq256_n256_local_bpe_1024",
)
def main(config):
    run_distillation(config)


if __name__ == "__main__":
    main()
