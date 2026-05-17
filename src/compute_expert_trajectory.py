import warnings

import hydra

from src.distillation.expert import run_expert_trajectory

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="expert_trajectory_local_bpe_1024",
)

def main(config):
    run_expert_trajectory(config)


if __name__ == "__main__":
    main()