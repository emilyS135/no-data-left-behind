import re
from argparse import Namespace
from time import time
from typing import Any, Mapping

import mlflow
from azure.ai.ml import MLClient
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from src.logging.azureml import setup_azure_ml_client
from src.utils import _convert_params, _flatten_dict, get_logger

log = get_logger(__name__)


# code mostly taken from pytorch-lightning and simplified.
class MlflowLogger:
    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, Any] | None = None,
        run_id: str | None = None,
    ):
        self._experiment_name = experiment_name
        self._experiment_id: str | None = None
        self._run_name = run_name
        self._run_id = run_id
        self.tags = tags
        self._initialized = False
        self._nested_runs: list[Run] = []

        self._azureml_client: MLClient = setup_azure_ml_client()
        azureml_workspace = self._azureml_client.workspaces.get(
            self._azureml_client.workspace_name
        )
        self._tracking_uri = azureml_workspace.mlflow_tracking_uri if azureml_workspace else None
        self._mlflow_client = MlflowClient(self._tracking_uri)

    @property
    def azure_client(self) -> MLClient:
        """AzureML client to query remote jobs, etc."""
        return self._azureml_client

    @property
    def experiment(self) -> "MlflowClient":
        r"""Actual MLflow object. To use MLflow features in your code do the following.

        Example::

            self.logger.experiment.some_mlflow_function()
        """

        if self._initialized:
            return self._mlflow_client

        # making sure it's set for both the functional and the OOP API
        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)
        else:
            raise Exception("Invalid Tracking URI")

        if self._run_id is not None:
            run = self._mlflow_client.get_run(self._run_id)
            self._experiment_id = run.info.experiment_id
            self._initialized = True
            return self._mlflow_client

        if self._experiment_id is None:
            expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
            if expt is not None:
                self._experiment_id = expt.experiment_id
            else:
                log.warning(
                    f"Experiment with name {self._experiment_name} not found. Creating it."
                )
                self._experiment_id = self._mlflow_client.create_experiment(
                    name=self._experiment_name
                )

        if self._run_id is None:
            if self._run_name is not None:
                self.tags = self.tags or {}

                from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

                if MLFLOW_RUN_NAME in self.tags:
                    log.warning(
                        f"The tag {MLFLOW_RUN_NAME} is found in tags. "
                        + "The value will be overridden by {self._run_name}."
                    )
                self.tags[MLFLOW_RUN_NAME] = self._run_name

            run = self._mlflow_client.create_run(
                experiment_id=self._experiment_id,  # type: ignore
                tags=resolve_tags(self.tags),
            )
            self._run_id = run.info.run_id
        self._initialized = True
        return self._mlflow_client

    @property
    def run_id(self) -> str:
        """Create the experiment if it does not exist to get the run id.

        Returns:
            The run id.
        """
        _ = self.experiment
        return self._run_id  # type: ignore

    @property
    def experiment_id(self) -> str:
        """Create the experiment if it does not exist to get the experiment id.

        Returns:
            The experiment id.
        """
        _ = self.experiment
        return self._experiment_id  # type: ignore

    def log_hyperparams(self, params: dict[str, Any] | Namespace) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)

        from mlflow.entities import Param

        # Truncate parameter values to 500 characters.
        params_list = [Param(key=k, value=str(v)[:500]) for k, v in params.items()]

        # Log in chunks of 100 parameters (the maximum allowed by MLflow).
        for idx in range(0, len(params_list), 100):
            self.experiment.log_batch(run_id=str(self.run_id), params=params_list[idx : idx + 100])

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        from mlflow.entities import Metric

        metrics_list: list[Metric] = []

        timestamp_ms = int(time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                log.warning(f"Discarding metric with string value {k}={v}.")
                continue

            new_k = re.sub("[^a-zA-Z0-9_/. -]+", "", k)
            if k != new_k:
                RuntimeWarning(
                    "MLFlow only allows '_', '/', '.' and ' ' special characters in metric name."
                    + f" Replacing {k} with {new_k}."
                )
                k = new_k
            metrics_list.append(Metric(key=k, value=v, timestamp=timestamp_ms, step=step or 0))
        self.experiment.log_batch(run_id=str(self.run_id), metrics=metrics_list)

    def add_tags(self, tags: dict) -> None:
        for key, value in tags.items():
            self.experiment.set_tag(run_id=str(self.run_id), key=key, value=value)

    def log_file(self, path: str):
        self.experiment.log_artifact(run_id=str(self.run_id), local_path=path)

    def log_omegaconf(self, cfg: DictConfig):
        config_dict = OmegaConf.to_container(cfg)
        self.experiment.log_dict(
            run_id=str(self.run_id),
            dictionary=config_dict,  # type: ignore -> we only expect cfg to be converted into dicts
            artifact_file="config.yaml",
        )

    @property
    def experiment_name(self) -> str:
        """Get the experiment name.

        Returns:
            The experiment name.
        """
        return self._experiment_name

    def finalize(self, status: str = "success") -> None:
        if not self._initialized:
            return
        if status == "success":
            status = "FINISHED"
        elif status == "failed":
            status = "FAILED"
        elif status == "finished":
            status = "FINISHED"

        if self.experiment.get_run(str(self.run_id)):
            self.experiment.set_terminated(str(self.run_id), status)
