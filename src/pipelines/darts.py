from src.utils import get_logger
from darts.models import LightGBMModel
from itertools import product
from src.data import utils as utils
from src.data.utils import darts_series_from_df
from src.data.utils import inv_transform

import pandas as pd
import torch
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import (
    TorchForecastingModel,
    MixedCovariatesTorchModel,
)
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame
from pytorch_lightning.callbacks import Callback, EarlyStopping
from ray import tune
from ray.train import RunConfig
from ray.tune.result_grid import ResultGrid
from sklearn.preprocessing import StandardScaler

from darts.metrics.metrics import mae
from abc import ABC, abstractmethod

import os
from pathlib import Path
from typing import Any, Sequence, cast

log = get_logger(__name__)


class DataPreprocessor:
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.df = df

        self._scaler = StandardScaler()
        self._scaler.set_output(transform="pandas")

    def get_train_test_unscaled(self):
        test_ratio_or_date = (
            self.cfg.evaluation.test_start_date
            if "test_start_date" in self.cfg.evaluation
            else self.cfg.evaluation.test_ratio
        )
        train_set, test_set = utils.split_train_test(self.df, test_ratio_or_date)
        return train_set, test_set

    def _fit_transform_scaler(self, train_set) -> pd.DataFrame:
        return cast(pd.DataFrame, self._scaler.fit_transform(train_set))

    def get_scaler(self):
        return self._scaler

    def _scale_data(self, data) -> pd.DataFrame:
        return cast(pd.DataFrame, self._scaler.transform(data))

    def _create_darts_series(self, data):
        series, past_covariates, future_covariates = darts_series_from_df(
            data,
            self.cfg.dataset.freq,
            self.cfg.model.get("hist_exog_list", None),
            self.cfg.model.get("futr_exog_list", None),
        )
        return series, past_covariates, future_covariates

    def get_timeseries(self) -> dict[str, dict[str, TimeSeries | None]]:
        # --- split train / test set and standardize data

        train_df, test_df = self.get_train_test_unscaled()
        train_df = train_df.drop(columns="unique_id").set_index("ds")
        scaled_train_val_df = self._fit_transform_scaler(train_df).reset_index()

        unique_id = self.df[["ds", "unique_id"]]
        scaled_full_df = self.df.copy().drop(columns="unique_id").set_index("ds")
        scaled_full_df = self._scale_data(scaled_full_df).reset_index()  # type: ignore
        # add unique_id back to df since it is needed later
        scaled_full_df = pd.merge(scaled_full_df, unique_id, how="left", on=["ds"])

        self.test_start_date = test_df.iloc[0]["ds"]
        log.info(f"train+val split shape: {scaled_train_val_df.shape}")
        log.info(f"test split start: {self.test_start_date}")
        log.info(
            f"test split shape: {test_df.shape}, which is {len(test_df) / len(scaled_full_df):.1%} of the total dataset."
        )

        # train + val + test
        scaled_full_series, scaled_full_past_covariates, scaled_full_future_covariates = (
            self._create_darts_series(scaled_full_df)
        )

        # train + val
        tv_series, tv_past_covariates, tv_future_covariates = self._create_darts_series(
            scaled_train_val_df
        )
        # -- prepare validation split

        scaled_train_df, scaled_val_df = self.split_train_val(scaled_train_val_df)

        # split covariates from dataset

        train_series, train_past_covariates, train_future_covariates = self._create_darts_series(
            scaled_train_df
        )
        val_series, val_past_covariates, val_future_covariates = self._create_darts_series(
            scaled_val_df
        )

        data = {
            "train": {
                "series": train_series,
                "past_covariates": train_past_covariates,
                "future_covariates": train_future_covariates,
            },
            "val": {
                "series": val_series,
                "past_covariates": val_past_covariates,
                "future_covariates": val_future_covariates,
            },
            "train_val": {
                "series": tv_series,
                "past_covariates": tv_past_covariates,
                "future_covariates": tv_future_covariates,
            },
            "all": {
                "series": scaled_full_series,
                "past_covariates": scaled_full_past_covariates,
                "future_covariates": scaled_full_future_covariates,
            },
        }
        return data

    def split_train_val(self, tv_df):
        val_ratio_or_date = (
            pd.Timestamp(self.cfg.evaluation.val_start_date)
            if "val_start_date" in self.cfg.evaluation
            else self.cfg.evaluation.val_ratio
        )
        # darts separates training and validation set more strictly than NF.
        # we allow some overlap between train and val of size {lookback},
        # so the first prediction can start at the val-set start date.
        if isinstance(val_ratio_or_date, pd.Timestamp):
            delta = self.cfg.evaluation.lookback * pd.to_timedelta(self.cfg.dataset.freq)
            val_start_adj = val_ratio_or_date - delta
            val_set = tv_df[tv_df.ds >= val_start_adj]
            train_set = tv_df[tv_df.ds < val_ratio_or_date]
            val_set_size = len(val_set)
        else:
            val_set_size = round(val_ratio_or_date * len(tv_df.ds.unique()))
            val_set_size_adj = val_set_size - self.cfg.evaluation.lookback
            log.warning(
                f"using overlap of l={self.cfg.evaluation.lookback} between train and val set to allow for lookback of train as part of val"
            )
            val_set = tv_df.iloc[val_set_size_adj:]
            train_set = tv_df.iloc[:val_set_size]
            assert len(val_set) == val_set_size
        log.info(
            f"val split size: {val_set_size}, which starts from date {val_ratio_or_date} and is roughly {val_set_size / len(train_set):.1%} of the train split or {val_set_size / len(tv_df):.1%} of the total dataset."
        )
        log.warning(
            f"using overlap of l={self.cfg.evaluation.lookback} between train and val set to allow for lookback of train as part of val"
        )
        return train_set, val_set


class ModelTuner(ABC):
    @abstractmethod
    def refit_model(self, train, val, train_val, all_data) -> tuple[GlobalForecastingModel, dict]:
        pass


class MixedTorchModelTuner(ModelTuner):
    def __init__(self, cfg):
        self.cfg = cfg

    def refit_model(self, train, val, train_val, all_data) -> tuple[GlobalForecastingModel, dict]:
        """refits a LGBM untrained model on the whole dataset including the validation data

        Args:
            train: training data excluding validation data
            val: validation data
            train_val: training and validation data
            all_data : whole dataset including training, validation, and test set

        Returns:
            tuple[GlobalForecastingModel, dict]: refitted model with best config found by hpo, best config including best hyperparameters
        """
        best_config = self.ray_hyperparameter_optimization(train, val)
        model = instantiate(
            self.cfg.model.model,
            pl_trainer_kwargs={
                "callbacks": self._get_callbacks(retrain=True),
                "enable_progress_bar": self.cfg.verbose > 0,
                "devices": 1,  # do not use distributed training here
                "num_nodes": 1,
            },
            **best_config,
        )

        model.fit(
            series=train_val["series"],
            past_covariates=train_val["past_covariates"],
            future_covariates=train_val["future_covariates"],
            verbose=self.cfg.verbose > 0,
        )
        return model, best_config

    def ray_hyperparameter_optimization(self, train, val):
        """performs hyperparameter tuning for MixedCovariatesTorchModel models using ray tune

        Args:
            train (_type_): training data excluding validation set
            val (_type_): validation data

        Returns:
            best_config: best model config found by ray tune
        """

        hpo_results = self.tune(train, val)

        best_result = hpo_results.get_best_result(metric="val_loss", mode="min")
        best_config = best_result.config

        if best_config is None:
            raise TypeError("something went wrong during ray tune HPO. no best config.")
        log.info("Best hyperparameters found were: ", best_config)
        log.info(best_config)
        return best_config

    def _prepare_hpo_params(self):
        self.param_space = cast(
            dict[str, Any], OmegaConf.to_container(instantiate(self.cfg.model.config))
        )

        # use max 8 CPUs and 1 GPU per trial
        cpus = os.cpu_count()
        gpus = torch.cuda.device_count()
        if gpus > 0:
            device_dict = {"gpu": 1}
        else:
            if cpus is None:
                device_dict = {}
            else:
                # these are the resources used *per trial*. less resources = more parallel trials
                cpus = min(cpus, 8)
                device_dict = {"cpu": cpus}
        self.device_dict = device_dict
        self.tune_config = tune.TuneConfig(
            mode="min",
            metric="val_loss",
            num_samples=self.cfg.model.num_samples,
        )
        self.run_config = RunConfig(name=self.cfg.name, verbose=self.cfg.verbose, log_to_file=True)

    def _get_callbacks(self, retrain=False):
        callbacks = []
        try:
            if retrain:
                early_stopper: EarlyStopping = instantiate(
                    self.cfg.model.early_stopping,
                    check_on_train_epoch_end=True,
                    monitor="train_loss",
                )
            else:
                early_stopper: EarlyStopping = instantiate(self.cfg.model.early_stopping)
            callbacks.append(early_stopper)
        except Exception:
            pass
        try:
            tune_callback = instantiate(self.cfg.model.tune_report_callback)
            callbacks.append(tune_callback)
        except Exception:
            pass
        return callbacks

    def _get_training_function(self):
        return ray_tune_darts_train_fn

    def tune(
        self, train: dict[str, TimeSeries | None], val: dict[str, TimeSeries | None]
    ) -> ResultGrid:
        self._prepare_hpo_params()

        train_fn_with_parameters = tune.with_parameters(
            self._get_training_function(),
            callbacks=self._get_callbacks(),
            train=train["series"],
            val=val["series"],
            past_covariates=train["past_covariates"],
            future_covariates=train["future_covariates"],
            val_past_covariates=val["past_covariates"],
            val_future_covariates=val["future_covariates"],
            cfg=self.cfg,
        )

        tuner = tune.Tuner(
            tune.with_resources(train_fn_with_parameters, self.device_dict),  # type: ignore
            param_space=self.param_space,
            tune_config=self.tune_config,
            run_config=self.run_config,
        )

        log.info("start hyperparameter tuning.")
        results = tuner.fit()
        log.info("finished tuning.")

        return results


class LGBMModelTuner(ModelTuner):
    def __init__(self, cfg):
        self.cfg = cfg

    def refit_model(self, train, val, train_val, all_data) -> tuple[GlobalForecastingModel, dict]:
        """refits a LGBM untrained model on the whole dataset including the validation data

        Args:
            train: training data excluding validation data
            val: validation data
            train_val: training and validation data
            all_data : whole dataset including training, validation, and test set

        Returns:
            tuple[GlobalForecastingModel, dict]: refitted model with best config found by hpo, best config including best hyperparameters
        """
        # gridsearch to find best hyperparameter
        model_untrained, best_config = self.lgbm_gridsearch(train, val, all_data)
        model_untrained.fit(
            series=train_val["series"],
            past_covariates=train_val["past_covariates"],
            future_covariates=train_val["future_covariates"],
        )
        model = model_untrained
        return model, best_config

    def lgbm_gridsearch(self, train, val, all_data):
        """perfroms hyperparameter tuning for LGBM using simple grid search

        Args:
            train (_type_): training data excluding validation set
            val (_type_): validation data
            all_data (_type_): whole dataset, needed for future/past covariates

        Returns:
            model_untrained: untrained model with best config
            best_params: best model config found by gridsearch
        """
        param_space = cast(
            dict[str, Any], OmegaConf.to_container(instantiate(self.cfg.model.config))
        )
        param_space["output_chunk_length"] = [self.cfg.evaluation.horizon]
        num_samples = self.cfg.model.model.num_samples
        # compute all hyperparameter combinations from selection
        params_cross_product = len(list(product(*param_space.values())))
        n_samples = min(params_cross_product, num_samples)

        model_untrained, best_params, metric_score = LightGBMModel.gridsearch(
            parameters=param_space,
            series=train["series"],
            past_covariates=all_data["past_covariates"],
            future_covariates=all_data["future_covariates"],
            val_series=val["series"],
            n_jobs=-1,
            show_warnings=True,
            metric=mae,
            verbose=self.cfg.verbose > 0,
            n_random_samples=n_samples,
        )
        if best_params is None:
            raise TypeError("something went wrong during lgbm HPO. no best config.")
        log.info("Best hyperparameters found were: ", best_params)
        log.info(best_params)
        log.info("model")
        log.info(model_untrained)
        log.info("metric_score")
        log.info(metric_score)

        return model_untrained, best_params


class Evaluation:
    def __init__(self, cfg, model: GlobalForecastingModel, data_preprocessor: DataPreprocessor):
        self.cfg = cfg
        self.model = model
        self.data_preprocessor = data_preprocessor

    def historical_forecasts(self, data: dict[str, TimeSeries | None]) -> DataFrame:
        histfcst_timeseries = self.model.historical_forecasts(
            series=data["series"],
            past_covariates=data["past_covariates"],
            future_covariates=data["future_covariates"],
            start=self.data_preprocessor.test_start_date,
            last_points_only=False,
            forecast_horizon=self.cfg.evaluation.horizon,
            stride=self.cfg.evaluation.horizon,
            retrain=False,
            verbose=self.cfg.verbose > 0,
            show_warnings=True,
        )

        if isinstance(histfcst_timeseries, list):
            histfcst_df = pd.concat(
                [t.pd_dataframe() for t in histfcst_timeseries if isinstance(t, TimeSeries)]
            )
        else:
            histfcst_df = histfcst_timeseries.pd_dataframe()

        model_name = self.cfg.model.model_name

        # inverse transform the forecast to calculate metrics
        histfcst_df = inv_transform(self.data_preprocessor.get_scaler(), histfcst_df, col_name="y")
        histfcst_df = histfcst_df.rename(columns={"y": model_name})

        return histfcst_df

    def calculate_metrics(self, histfcst_df):
        model_name = self.cfg.model.model_name
        # the forecast includes 90*24 values whereas the test_set includes 90*24+1. The last value has to be removed

        _, test_df = self.data_preprocessor.get_train_test_unscaled()
        # adjust test set size such that it's divisible by the horizon length
        rows_to_remove = len(test_df) % self.cfg.evaluation.horizon
        if rows_to_remove > 0:
            df_test_eval = test_df.iloc[:-rows_to_remove]
        else:
            df_test_eval = test_df

        # check if adjusted dataframe has the correct test data start and correct end date and length
        assert df_test_eval["ds"].iloc[0] == pd.to_datetime(
            test_df["ds"].iloc[0]
        ), "The test set's start date is not correct."
        assert df_test_eval["ds"].iloc[-1] == pd.to_datetime(test_df["ds"].iloc[0]) + pd.Timedelta(
            hours=(len(test_df)) - 1 - rows_to_remove
        ), "The last entry does not correspond to the expected date."
        assert (
            len(df_test_eval) % self.cfg.evaluation.horizon == 0
        ), "The number of test data points is not correct. Has to be divisible by the horizon length"

        labels = torch.tensor(df_test_eval["y"].values)
        forecasts = torch.tensor(histfcst_df[model_name].values)

        # calculate all metrics at once.
        metric_collection = instantiate(self.cfg.model.model.torch_metrics)

        metrics_dict = metric_collection(forecasts, labels)

        # parse tensors into dataframe
        metrics_dict = {k: v.numpy() for k, v in metrics_dict.items()}
        df_metrics: pd.DataFrame = pd.DataFrame.from_dict(
            metrics_dict, orient="index", columns=[model_name]
        ).reset_index(names=["metric"])

        return df_metrics


# ----


def ray_tune_darts_train_fn(
    model_args: dict,
    callbacks: Sequence[Callback],
    train: TimeSeries,
    val: TimeSeries | None,
    past_covariates: TimeSeries | None,
    future_covariates: TimeSeries | None,
    val_past_covariates: TimeSeries | None,
    val_future_covariates: TimeSeries | None,
    cfg: DictConfig,
):
    """
    Instantiate and fit a Darts model. Used as the training function in ray tune hyperparameter optimization.

    Args:
    model_args (dict): model hyperparameters, provided by ray tune.
    callbacks (Sequence[pl.Callback]): A list of Lightning callbacks for the model training.
    train (TimeSeries): The training data.
    val (TimeSeries): The validation data.
    past_covariates (pd.DataFrame): Past covariates for the training data.
    future_covariates (pd.DataFrame): Future covariates for the training data.
    val_past_covariates (pd.DataFrame): Past covariates for the validation data.
    val_future_covariates (pd.DataFrame): Future covariates for the validation data.
    cfg (DictConfig): The configuration object.

    """
    # Create the model by joining base config and selected model_args from Ray Tune
    model: LightGBMModel | TorchForecastingModel = instantiate(
        cfg.model.model,
        pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": False},
        **model_args,
    )
    model.fit(
        series=train,
        val_series=val,
        val_past_covariates=val_past_covariates,
        val_future_covariates=val_future_covariates,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
    )


def new_darts_pipeline(
    cfg: DictConfig, model: LightGBMModel | MixedCovariatesTorchModel, kwargs: dict
):
    if cfg.evaluation.cv and cfg.evaluation.retrain:
        raise NotImplementedError("retrain cv not implemented for darts.")

    # --- Prep Data
    df = kwargs["df"]
    data_preprocessor = DataPreprocessor(cfg, df)
    data = data_preprocessor.get_timeseries()

    train = data["train"]
    val = data["val"]
    train_val = data["train_val"]
    all_data = data["all"]

    # --- Hyperparameter Optimization

    if isinstance(model, LightGBMModel):
        model_trainer = LGBMModelTuner(cfg)

    elif isinstance(model, MixedCovariatesTorchModel):
        model_trainer = MixedTorchModelTuner(cfg)
    else:
        raise ValueError("unknown model type.")
    # --- tune and refit on whole training set
    model, best_config = model_trainer.refit_model(train, val, train_val, all_data)

    kwargs["best_config"] = best_config
    utils.save_yaml(kwargs["best_config"], Path.cwd(), f"best_hpo_config_{cfg.name}", "hpo")

    # --- Evaluate on Test Set
    evaluation = Evaluation(cfg, model, data_preprocessor)

    forecasts = evaluation.historical_forecasts(all_data)
    log.info("historical_forecasts output:")
    log.info("\n" + forecasts.head().to_string())
    log.info("...")
    log.info("\n" + forecasts.tail().to_string())
    metrics = evaluation.calculate_metrics(forecasts)
    log.info("metrics:")
    log.info("\n" + metrics.head().to_string())

    return forecasts, metrics
