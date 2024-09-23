import logging
import random
import warnings
from neuralforecast.auto import tune

import numpy as np
from omegaconf import DictConfig
import torch

from src.utils import rich_utils

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    return logger


log = get_logger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=False, save_to_file=False)


# adapted and simplified from pytorch-lightning
def seed_everything(seed: int | None = None) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    Args:
        seed: the integer value seed for global random state.
            If `None`, will select it randomly.
    """
    if seed is None:
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
        log.warning(f"No seed found, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        log.warning(
            f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    log.info(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed


def tied_sampler(config, config_key):
    # returns whatever is in tune config at [key] to tie the variables together
    return config[config_key]


def resolve_tune(target_dict: dict):
    """Resolves nested 'tune' dictionaries in the target dictionary
    by converting it to individual key value pairs of parameter: tune_class
    and placing them directly on the target dictionary.
    e.g. {tune: { choices: {param_1: [1, 2, 3], param_1: [0.001, 0.01]}}
    -> {param_1: tune.choice (with values [1, 2, 3]), param_1: tune.choice ([0.001, 0.01])}

    Args:
        target_dict (dict):
            target root dictionary that contains a nested 'tune' dictionary with parameters and lists for tune classes.

    """
    for tune_method, params in target_dict.get("tune", {}).items():
        for param, tune_values in params.items():
            _function = getattr(tune, tune_method)
            target_dict[param] = eval(f"_function({tune_values})")
    target_dict.pop("tune")


def _select_seed_randomly(
    min_seed_value: int = min_seed_value, max_seed_value: int = max_seed_value
) -> int:
    return random.randint(min_seed_value, max_seed_value)

