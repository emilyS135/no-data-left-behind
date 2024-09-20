from src.data import utils as utils
from src.data.datamodule import DataModule

from src.data.ltsf_datamodule import LTSFDataModule

from src.logging.mlflow import MlflowLogger
from src.evaluation import get_metric
from src.pipelines import with_mlflow
from src.pipelines.darts import new_darts_pipeline
from src.pipelines.neuralforecast import neuralforecast_pipe
from src.utils.types import MetricsDict
from src.utils import get_logger
from src.visualization.visualize import catplot_by_station_and_measure

import pandas as pd
import mlflow
import pytorch_lightning as pl
from darts.models import LightGBMModel
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from omegaconf import DictConfig


from pathlib import Path
from typing import Any, cast

log = get_logger(__name__)


@with_mlflow
def run_ltsf_pipe(
    cfg: DictConfig,
    datamodule: DataModule,
    model,
    logger: MlflowLogger | None,
) -> tuple[pd.DataFrame, pd.DataFrame] | MetricsDict | pd.DataFrame | None:
    # configure automatic logging
    if logger:
        if isinstance(model, pl.LightningModule):
            mlflow.pytorch.autolog(
                checkpoint=False, log_models=False, exclusive=False, log_datasets=False
            )
        else:
            mlflow.autolog(
                log_input_examples=True, log_models=False, exclusive=False, log_datasets=False
            )
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
        ret = neuralforecast_pipe(cfg, model, metrics, data_kwargs, logger)
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

    if logger:
        g = catplot_by_station_and_measure(eval_test)
        logger.experiment.log_figure(
            run_id=str(logger.run_id), figure=g.figure, artifact_file="catplot_test_eval.png"
        )

    log.info(f"Train set eval:\n{train_summary.to_string()}")
    log.info(f"Test set params: horizon: {eval_cfg.horizon}")
    log.info(f"Test set eval:\n{test_summary.to_string()}")

    return train_summary, test_summary


def log_ltsf_results(
    ret: MetricsDict | tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame, logger
) -> MetricsDict:
    def metric_df_to_dict(df, split: str):
        df["metric"] = split + "_" + df["metric"]
        return df.set_index("metric").iloc[:, 0].to_dict()

    if isinstance(ret, tuple):
        forecasts, metrics = ret
        # extract metrics from forecasts
        # Flatten the DataFrame so that each metric name and value gets its own row
        test_dict = metric_df_to_dict(metrics, "total")

        if logger:
            if isinstance(logger, MlflowLogger):
                utils.assert_feature_existence(forecasts, ["ds"])
                from mlflow.entities import Metric as MlflowMetric

                forecasts = forecasts.select_dtypes(exclude=["object"])
                try:
                    logger.experiment.log_table(logger.run_id, forecasts, "forecasts.json")
                except Exception as e:
                    log.debug(e)

                df_melted = forecasts.melt(id_vars=["ds"], var_name="metric", value_name="value")
                # Convert 'ds' to UNIX timestamp in milliseconds
                df_melted["timestamp"] = (df_melted["ds"].astype(int) / 10**6).astype(int)
                # "step" will just be timestamp, so it works in mlflow
                df_melted["step"] = df_melted["timestamp"]

                # Create MLflow Metric objects using a list comprehension
                metrics = [
                    MlflowMetric(
                        key=row["metric"],
                        value=pd.to_numeric(row["value"]),
                        timestamp=row["timestamp"],
                        step=row["step"],
                    )
                    for row in df_melted.to_dict("records")
                ]
                logger.experiment.log_batch(run_id=logger.run_id, metrics=metrics)

            logger.log_metrics(test_dict)
        return test_dict
    else:
        cv_dict = metric_df_to_dict(ret, "cv")
        if logger:
            logger.log_metrics(cv_dict)
        return cv_dict
