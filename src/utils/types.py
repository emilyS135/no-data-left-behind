from typing import Any, Protocol
import numpy as np
import pandas as pd

class MetricCallable(Protocol):
    def __call__(
        self,
        y: np.ndarray | pd.Series,
        y_hat: np.ndarray | pd.Series,
        weights: np.ndarray | None = None,
        axis: int | None = None,
        **kwargs,
    ) -> float | np.ndarray: ...


MetricsDict = dict[str, Any]
TrainTestMetricsTuple = tuple[MetricsDict, MetricsDict]
