import os
import sys

import hydra
from omegaconf import DictConfig

from src import utils
from src.pipelines import train_ltsf_pipeline


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig) -> None:
    # Applies optional utilities
    utils.extras(cfg)
    train_ltsf_pipeline(cfg)

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    print(sys.path)
    print(os.getcwd())
    main()
