

from neuralforecast.auto import BaseAuto
from omegaconf import DictConfig

from src.utils import get_logger, resolve_tune, seed_everything

from src.utils.types import MetricsDict, TrainTestMetricsTuple

from src.data import utils as utils
from src.data.datamodule import DataModule
from hydra.utils import instantiate


log = get_logger(__name__)



def train_ltsf_pipeline(cfg: DictConfig) -> TrainTestMetricsTuple | MetricsDict | None:
    from src.pipelines.ltsf import run_ltsf_pipe, log_ltsf_results

    datamodule, model = resolve_config(cfg)

    ret = run_ltsf_pipe(cfg=cfg, datamodule=datamodule, model=model)
    log.info("finished training")
    if ret is not None:
        ltsf_metrics = log_ltsf_results(ret)
        return ltsf_metrics
    else:
        return None


## helper -- could be moved


def resolve_config(cfg):
    log.info("Loaded Config:", cfg)

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed)

    # initiate dataset specific DataModule Preprocessor
    log.info(f"Instantiating datamodule <{cfg.dataset.datamodule._target_}>")
    datamodule: DataModule = instantiate(cfg.dataset.datamodule)

    log.info(f"Instantiating model <{cfg.model.model._target_}>")

    if "early_stopping" in cfg.model:
        early_stopping = instantiate(cfg.model.early_stopping)
        pl_trainer_kwargs = {"callbacks": [early_stopping]}
        model = instantiate(cfg.model.model, _convert_="partial", pl_trainer_kwargs=pl_trainer_kwargs)
    else:
        model = instantiate(cfg.model.model, _convert_="partial")

    # found no way to declare the tune._ classes for auto tuning from the yaml itself,
    # so we have to do it manually here.
    if isinstance(model, BaseAuto):
        resolve_tune(model.config)  # type: ignore

    return datamodule, model

