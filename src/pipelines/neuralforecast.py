import mlflow
import numpy as np
from statsforecast import StatsForecast
from tqdm.auto import tqdm
from utilsforecast.evaluation import evaluate
from typing import cast
from src.data import utils as utils
from src.logging.mlflow import MlflowLogger
from src.utils.types import MetricCallable, MetricsDict
from src.utils import get_logger
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from neuralforecast.auto import BaseAuto
from neuralforecast.core import NeuralForecast, _insample_times
from omegaconf import DictConfig
from pandas import DataFrame

log = get_logger(__name__)


def cross_validation_with_retraining(
    forecaster: NeuralForecast,
    cfg: DictConfig,
    metrics: dict[str, MetricCallable],
    kwargs: dict,
    logger: MlflowLogger | None,
) -> tuple[DataFrame, DataFrame]:
    """
    Retrain the model using a growing window cross validation and evaluate its performance.

    Parameters:
    forecaster (NeuralForecast): the forecaster to be retrained and evaluated
    cfg (DictConfig): configuration for the retraining and evaluation
    metrics (dict[str, MetricCallable]): metrics to be used for evaluation
    kwargs (dict): additional keyword arguments
    logger (MlflowLogger | None): optional logger for logging metrics and hyperparameters

    Returns:
    tuple[DataFrame, DataFrame]: a tuple of two DataFrames, the first containing the forecasts and the second containing the aggregated evaluation metrics
    """

    horizon = cfg.evaluation.horizon
    freq = cfg.dataset.freq

    model_name = "unknown_model"

    forecasts = {}
    metric_results = {}
    log.warning("this might not work as intended anymore. need to set step_size")
    step_size = horizon
    log.info("starting re-training using growing window.")
    for split, (retrain_set, test_set) in enumerate(
        tqdm(
            utils.get_walk_forward_splits(
                kwargs["df"],
                kwargs["test_start"],
                horizon=horizon,
                step_size=step_size,
                n_splits=kwargs["n_windows"],
                gap=0,  # could be horizon
            )
        )
    ):
        if isinstance(logger, MlflowLogger):
            parent_run = logger.experiment.get_run(logger.run_id)
            if parent_run is not None:
                parent_run_name = parent_run.info.run_name
                parent_run_id = parent_run.info.run_id
            else:
                parent_run_name = "unknown_parent_run"
                parent_run_id = "unknown_parent_id"
            child_run_name = f"{parent_run_name}_{split}"
            with mlflow.start_run(
                run_name=child_run_name, experiment_id=logger.experiment_id, nested=True
            ) as child_run:
                log.info(
                    f"starting nested mlflow run {child_run_name} with id {child_run.info.run_id} of experiment {logger.experiment_name} under parent {parent_run_name} with id {parent_run_id}"
                )

                model_name, fcsts_labelled, cv_results = train_eval_cross_validation_fold(
                    forecaster,
                    cfg,
                    metrics,
                    horizon,
                    freq,
                    split,
                    retrain_set,
                    test_set,
                )
                logger.experiment.log_table(
                    child_run.info.run_id, cv_results, f"cv_split_{split}.json"
                )

        else:
            model_name, fcsts_labelled, cv_results = train_eval_cross_validation_fold(
                forecaster, cfg, metrics, horizon, freq, split, retrain_set, test_set
            )

        metric_results[split] = cv_results
        forecasts[split] = fcsts_labelled

    all_forecasts = pd.concat([df for df in forecasts.values()])

    aggregated_metrics = cast(
        DataFrame,
        evaluate(
            all_forecasts,
            metrics=list(metrics.values()),
            train_df=None,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            models=[model_name],
        ),
    )
    aggregated_metrics = aggregated_metrics.drop(columns=["unique_id"])

    return all_forecasts, aggregated_metrics


def cross_validation_evaluation(
    forecaster: NeuralForecast,
    cfg: DictConfig,
    metrics: dict[str, MetricCallable],
    kwargs: dict,
    logger: MlflowLogger | None,
) -> tuple[DataFrame, DataFrame]:
    """
    Evaluates the performance of a forecaster using cross-validation without re-training.

    Parameters:
    forecaster (NeuralForecast): the forecaster to be evaluated
    cfg (DictConfig): configuration for the evaluation
    metrics (dict[str, MetricCallable]): metrics to be used for evaluation
    kwargs (dict): additional keyword arguments
    logger (MlflowLogger | None): optional logger for logging metrics and hyperparameters

    Returns:
    tuple[DataFrame, DataFrame]: a tuple of two DataFrames, the first containing the forecasts and the second containing the evaluation metrics
    """
    # use fixed training set and step over test set to evaluate
    horizon = cfg.evaluation.horizon

    log.info(f"evaluating {forecaster.models[0]} using cross validation.")

    forecasts = {}
    metric_results = {}
    model = forecaster.models[0]
    model_name = str(model)

    for split, (fcst_input, fcst_labels) in enumerate(
        tqdm(
            utils.get_walk_forward_splits(
                kwargs["df"],
                kwargs["test_start"],
                horizon=horizon,
                step_size=horizon,
                test_size=horizon,
                gap=0,  # could be horizon
            )
        )
    ):
        log.info(
            f"split {split}: total input size: {fcst_input.shape}\t labelled fcst size: {fcst_labels.shape}"
        )

        # we re-use the fitted model end only evaluate on each fcst split.
        # the "train" split will be used as historic input to predict the next horizon, the "test" split as future covariates and afterwards as labels

        # make sure the exogenous variables are actually used
        if len(model.futr_exog_list) <= 0:
            log.warning(f"no future exog. variables are used by model {model_name}!")
        if len(model.hist_exog_list) <= 0:
            log.warning(f"no historic exog. variables are used by model {model_name}!")
            # we might have some test splits that are too long considering stride

        predict_kwargs = {
            # "df": df_test,
            "static_df": kwargs.get("static_df", None),
            "futr_df": kwargs.get("futr_df", fcst_labels.drop(columns=["y"])),
            "verbose": cfg.verbose > 0,
            # "step_size": step_size,
        }
        # if, and only if, test_set_size == horizon, we can use nf.predict(). otherwise, we can't.
        if len(fcst_labels) != horizon:
            log.warning(
                f"test split length ({len(fcst_labels)}) does not match horizon ({horizon})"
            )
        # Predict
        model_fcsts = cast(
            DataFrame,
            forecaster.predict(fcst_input, **predict_kwargs),
        )
        print(f"model fcsts shape: {model_fcsts.shape}")
        # concat label & predictions
        fcsts_and_labels = pd.merge(model_fcsts, fcst_labels, how="left", on=["unique_id", "ds"])
        # calculate metrics
        cv_results = cast(
            DataFrame,
            evaluate(
                fcsts_and_labels,
                metrics=list(metrics.values()),
                train_df=fcst_input,
                id_col="unique_id",
                time_col="ds",
                target_col="y",
                models=[model_name],
            ),
        )
        cv_results["split"] = split
        fcsts_and_labels["split"] = split
        log.info(f"metrics for split {split}:")
        log.info("\n" + cv_results.to_string())

        metric_results[split] = cv_results
        forecasts[split] = fcsts_and_labels

    all_forecasts = pd.concat([df for df in forecasts.values()])
    try:
        Y_df = all_forecasts[["ds", "y", "unique_id"]]
        fcst_df = all_forecasts.loc[:, list(all_forecasts.columns.difference(["y"]))]
        fig = StatsForecast.plot(
            Y_df,
            fcst_df,
            engine="matplotlib",
            plot_random=False,
        )
        if logger:
            curr_date = pd.Timestamp.now().strftime("%Y%m%d_%H-$M-$S")
            exp_name = f"{model_name}_dropout={model.config.dropout}_input_size={model.config.input_size}_learning_rate={model.config.learning_rate}_{curr_date}"
            logger.experiment.log_figure(logger.run_id, fig, f"testset_forecasts_{exp_name}.png")
    except Exception as e:
        log.exception(e)

    aggregated_metrics = cast(
        DataFrame,
        evaluate(
            all_forecasts,
            metrics=list(metrics.values()),
            train_df=None,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            models=[model_name],
        ),
    )
    aggregated_metrics = aggregated_metrics.drop(columns=["unique_id"])
    return all_forecasts, aggregated_metrics


def neuralforecast_pipe(
    cfg: DictConfig,
    model: pl.LightningModule,
    metrics: dict[str, MetricCallable],
    kwargs: dict,
    logger: MlflowLogger | None,
) -> MetricsDict | DataFrame | None | tuple[DataFrame, DataFrame]:
    """
    This function is the main entry point for the NeuralForecast pipeline.
    It takes in a configuration, a model, metrics, keyword arguments, and a logger (optional).
    Depending on the configuration, it will either perform hyperparameter tuning (HPO) with cross-validation (CV) evaluation,
    or cross-validation evaluation without re-training the model.

    Parameters:
    cfg (DictConfig): the main configuration for the pipeline
    model (pl.LightningModule): the model to be used for forecasting
    metrics (dict[str, MetricCallable]): a dictionary of metrics to be used for evaluation
    kwargs (dict): additional keyword arguments
    logger (MlflowLogger | None): an optional logger for logging metrics and hyperparameters

    Returns:
    MetricsDict | DataFrame | None | tuple[DataFrame, DataFrame]:
    depending on the configuration, it will return either a dictionary of metrics,
    a pandas DataFrame, None, or a tuple of two DataFrames
    """

    # Create a NeuralForecast instance with the given model

    forecaster = NeuralForecast(models=[model], **cfg.model.neuralforecast)
    kwargs.pop("h", None)  # unsupported

    # --- prepare dataset splits ---

    df: DataFrame = kwargs["df"]
    try:
        test_ratio_or_date = cfg.evaluation.test_start_date
    except Exception:
        test_ratio_or_date = cfg.evaluation.test_ratio
    try:
        val_ratio_or_date = pd.Timestamp(cfg.evaluation.val_start_date)
    except Exception:
        val_ratio_or_date = cfg.evaluation.val_ratio

    df_train, df_test = utils.split_train_test(df, test_ratio_or_date)
    test_start_date = df_test.iloc[0]["ds"]
    kwargs["test_start"] = test_start_date
    log.info(f"train+val split shape: {df_train.shape}")
    log.info(f"test split start: {test_start_date}")
    log.info(
        f"test split shape: {df_test.shape}, which is {len(df_test) / len(df):.1%} of the total dataset."
    )
    if isinstance(val_ratio_or_date, pd.Timestamp):
        val_set = df_train[df_train.ds >= val_ratio_or_date]
        val_set_size = len(val_set)
        log.info(
            f"val split size: {val_set_size}, which starts from date {val_ratio_or_date} and is {val_set_size / len(df_train):.1%} of the train split or {val_set_size / len(df):.1%} of the total dataset."
        )

    else:
        val_set_size = round(val_ratio_or_date * len(df_train.ds.unique()))
        log.info(
            f"val split size: {val_set_size}, which is {val_ratio_or_date:.1%} of the train split or {val_set_size / len(df):.1%} of the total dataset."
        )

    # --- run cross validation while iteratively re-training the model ---
    if cfg.evaluation.cv and cfg.evaluation.retrain:
        kwargs["n_windows"] = cfg.evaluation.n_windows
        forecaster = NeuralForecast(models=[model], **cfg.model.neuralforecast)
        return cross_validation_with_retraining(forecaster, cfg, metrics, kwargs, logger=logger)

    # --- or use regular split and optionally Hyperparameter Optimization ---
    else:
        # Fit the model with the provided validation set size
        fit_args = dict(verbose=cfg.verbose > 0, val_size=val_set_size)
        forecaster.fit(df_train, **fit_args)
        if isinstance(model, BaseAuto):
            log.info("finished HPO. results:")
            hpo_results = forecaster.models[0].results.get_dataframe()
            log.info("\n" + hpo_results.to_string())
            log.info("best config:")
            kwargs["best_config"] = forecaster.models[0].results.get_best_result().config
            log.info(kwargs["best_config"])
            # should be logged by mlflow/lightning, but to be sure.
            utils.save_yaml(kwargs["best_config"], Path.cwd(), f"best_hpo_config_{model}", "hpo")
            if logger is not None:
                logger.log_hyperparams(kwargs["best_config"])

        # --- use Cross-Validation for Model Evaluation
        if cfg.evaluation.cv and not cfg.evaluation.retrain:
            # re-use forecaster
            return cross_validation_evaluation(forecaster, cfg, metrics, kwargs, logger=logger)


def train_eval_cross_validation_fold(
    forecaster: NeuralForecast,
    cfg,
    metrics: dict[str, MetricCallable],
    horizon: int,
    freq: str,
    split: int,
    retrain_set: DataFrame,
    test_set: DataFrame,
) -> tuple[str, DataFrame, DataFrame]:
    """
    Calculate cross-validation metrics by fitting a new model for a CV fold and doing predictions for all possible cutoff dates inside the test set of this fold.

    Parameters:
    forecaster (NeuralForecast): The NeuralForecast instance to use.
    cfg: The configuration for the experiment.
    metrics (dict[str, MetricCallable]): A dictionary of metrics to use for evaluation.
    horizon (int): The forecast horizon.
    freq (str): The frequency of the data.
    split (int): The current split number.
    retrain_set (DataFrame): The retraining dataset.
    test_set (DataFrame): The test dataset.

    Returns:
    tuple[str, DataFrame, DataFrame]: A tuple containing the model name, the labeled forecast DataFrame, and the cross-validation results DataFrame.
    """
    log.info(f"split {split}: train size: {retrain_set.shape}\t test size: {test_set.shape}")

    # now we fit a new model for each split and do predictions for all possible cutoff dates inside the test split
    validation_set_size = 0
    # recreate everything, just to be sure
    forecaster.fit(
        retrain_set,
        val_size=validation_set_size,
        verbose=cfg.verbose > 0,
        use_init_models=True,  # resets models to be sure
    )

    # now we do *NOT* use the forecaster anymore, but the model itself, because we do CV differently.
    # make sure to take the fitted model
    model = forecaster.models[0]
    model_name = str(model)
    step_size = model.input_size
    # make sure the exogenous variables are actually used
    if len(model.futr_exog_list) <= 0:
        log.warning(f"no future exog. variables are used by model {model_name}!")
    if len(model.hist_exog_list) <= 0:
        log.warning(f"no historic exog. variables are used by model {model_name}!")
        # we might have some test splits that are too long considering stride

    n, rm = np.divmod(len(test_set) - horizon, step_size)
    if rm != 0:
        log.warn(f"test split size is not dividable by step size, cutting off {rm} samples")
        test_set = test_set.iloc[:-rm]

    # yup, using private methods here..
    trimmed_dataset, uids, last_dates, ds = forecaster._prepare_fit(
        df=test_set,
        static_df=None,
        sort_df=True,
        predict_only=True,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    # Generate dataframe with input dates and cutoffs
    fcsts_timesteps_df = _insample_times(
        times=ds,
        uids=uids,
        indptr=trimmed_dataset.indptr,
        h=horizon,
        freq=freq,
        step_size=step_size,
        id_col="unique_id",
        time_col="ds",
    )
    fcsts_timesteps_df = cast(DataFrame, fcsts_timesteps_df)
    # forecast for all possible cut-off dates
    # Test size is the number of periods to forecast (full size of trimmed dataset)
    model.set_test_size(test_size=trimmed_dataset.max_size)

    model_fcsts = model.predict(trimmed_dataset, step_size=step_size)
    # model.set_test_size(test_size=test_size)  # Set original test_size
    print(f"model fcsts shape: {model_fcsts.shape}")
    model_fcsts_df = DataFrame(model_fcsts, columns=[model_name])

    # get labels
    df_len_min = min(len(fcsts_timesteps_df), len(model_fcsts_df))
    fcst_df_trimmed = fcsts_timesteps_df.iloc[:df_len_min]
    fcsts_timestamped = pd.concat([fcst_df_trimmed, model_fcsts_df], axis=1)

    n_out = fcst_df_trimmed.ds.nunique()
    y_out = DataFrame(
        {
            "unique_id": np.repeat(uids, n_out),  # type: ignore
            "ds": ds[:n_out],
            "y": trimmed_dataset.temporal[:n_out, 0].numpy(),
        }
    )
    # concat label & predictions
    fcsts_labelled = pd.merge(fcsts_timestamped, y_out, how="left", on=["unique_id", "ds"])
    # calculate metrics
    cv_results = cast(
        DataFrame,
        evaluate(
            fcsts_labelled,
            metrics=list(metrics.values()),
            train_df=retrain_set,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            models=[model_name],
        ),
    )
    cv_results["split"] = split
    fcsts_labelled["split"] = split
    log.info(f"metrics for split {split}:")
    log.info("\n" + cv_results.to_string())
    return model_name, fcsts_labelled, cv_results
