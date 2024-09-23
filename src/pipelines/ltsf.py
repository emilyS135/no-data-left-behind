from src.data import utils as utils
from src.data.datamodule import DataModule

from src.data.ltsf_datamodule import LTSFDataModule

from src.evaluation import get_metric
from src.pipelines.darts import new_darts_pipeline
from src.pipelines.neuralforecast import neuralforecast_pipe
from src.utils.types import MetricsDict
from src.utils import get_logger

import pandas as pd
import pytorch_lightning as pl
from darts.models import LightGBMModel
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from omegaconf import DictConfig


from pathlib import Path
from typing import Any, cast

log = get_logger(__name__)


def run_ltsf_pipe(
    cfg: DictConfig,
    datamodule: DataModule,
    model,
) -> tuple[pd.DataFrame, pd.DataFrame] | MetricsDict | pd.DataFrame | None:
    # prepare dataset
    datamodule = cast(LTSFDataModule, datamodule)
    Y_df = datamodule.pipeline()
    futr_df = datamodule.futr_exog
    utils.assert_feature_existence(Y_df, ["y", "ds", "unique_id"])

    # ---
    # prepare the experiment configuration
    eval_cfg = cfg.evaluation
    metrics = {k: get_metric(k) for k in eval_cfg.metrics}
    data_kwargs: dict[str, Any] = dict(
        df=Y_df,
        verbose=cfg.verbose > 0,
        h=eval_cfg.horizon,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        futr_df=futr_df,
        static_df=None,
    )
    # run NeuralForecast or Darts fit / cross-validation pipeline
    if isinstance(model, pl.LightningModule):
        ret = neuralforecast_pipe(cfg, model, metrics, data_kwargs)
    elif isinstance(model, MixedCovariatesTorchModel):
        ret = new_darts_pipeline(cfg, model, data_kwargs)
    elif isinstance(model, LightGBMModel):
        ret = new_darts_pipeline(cfg, model, data_kwargs)
    else:
        raise ValueError("unknown model type.")
    if cfg.evaluation.cv and isinstance(ret, tuple):
        forecasts, cv_metrics = ret
        log.info("finished growing window cv. final results:")
        log.info("\n" + cv_metrics.to_string())
        utils.save_csv(forecasts, Path.cwd(), f"best_hpo_forecasts_{cfg.name}", "hpo")
        return forecasts, cv_metrics

    elif isinstance(ret, tuple):
        eval_train, eval_test = ret
    else:
        return None
    # aggregate results
    train_summary = eval_train.drop(columns="unique_id").groupby("metric").mean().reset_index()
    test_summary = eval_test.drop(columns="unique_id").groupby("metric").mean().reset_index()

    log.info(f"Train set eval:\n{train_summary.to_string()}")
    log.info(f"Test set params: horizon: {eval_cfg.horizon}")
    log.info(f"Test set eval:\n{test_summary.to_string()}")

    return train_summary, test_summary


def log_ltsf_results(
    ret: MetricsDict | tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame) -> MetricsDict:
    def metric_df_to_dict(df, split: str):
        df["metric"] = split + "_" + df["metric"]
        return df.set_index("metric").iloc[:, 0].to_dict()

    if isinstance(ret, tuple):
        forecasts, metrics = ret
        # extract metrics from forecasts
        # Flatten the DataFrame so that each metric name and value gets its own row
        test_dict = metric_df_to_dict(metrics, "total")

        return test_dict
    else:
        cv_dict = metric_df_to_dict(ret, "cv")
        return cv_dict
