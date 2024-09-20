import functools

import mlflow

from neuralforecast.auto import BaseAuto
from omegaconf import DictConfig

from src.utils import get_logger, resolve_tune, seed_everything

from src.utils.types import MetricsDict, TrainTestMetricsTuple

from ..data import utils as utils
from ..data.datamodule import DataModule
from ..logging.mlflow import MlflowLogger
from hydra.utils import instantiate


log = get_logger(__name__)


def with_mlflow(func):
    """Decorator that wraps (training) function and starts & finishes a new mlflow run.

    Use by adding @with_mlflow on top of function to be wrapped.

    Args:
        func (Callable): function to be wrapped

    Returns:
        Wrapped function
    """

    @functools.wraps(func)
    def mlflow_decorator(*args, **kwargs):
        logger = kwargs.get("logger")
        if isinstance(logger, MlflowLogger):
            with mlflow.start_run(logger.run_id):
                log.info(
                    f"starting mlflow run with id {logger.run_id} of experiment {logger.experiment_name}"
                )
                result = func(*args, **kwargs)
                logger.finalize()
            return result
        else:
            result = func(*args, **kwargs)
            return result

    return mlflow_decorator



def train_ltsf_pipeline(cfg: DictConfig) -> TrainTestMetricsTuple | MetricsDict | None:
    from src.pipelines.ltsf import run_ltsf_pipe, log_ltsf_results

    datamodule, model, _logger = resolve_config(cfg)

    ret = run_ltsf_pipe(cfg=cfg, datamodule=datamodule, model=model, logger=_logger)
    log.info("finished training")
    if ret is not None:
        ltsf_metrics = log_ltsf_results(ret, _logger)
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

    _logger = None
    if "logger" in cfg:
        log.info(f"Instantiating logger <{cfg.logger._target_}>")
        _logger = instantiate(cfg.logger)
        if isinstance(_logger, MlflowLogger):
            log.info("Logging hyperparameters")
            _logger.log_omegaconf(cfg)
        else:
            log.info("Unsupported Logger. No logger has been initialized.")
    else:
        log.info("No logger has been initialized.")
    return datamodule, model, _logger

