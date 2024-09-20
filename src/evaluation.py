import functools
from typing import Callable

import numpy as np

from src.utils import get_logger
from src.utils.types import MetricCallable
from numpy.typing import NDArray

log = get_logger(__name__)


def get_metric(name: str) -> MetricCallable:
    from utilsforecast import losses as utl

    try:
        scorer: MetricCallable = getattr(utl, name)
    except KeyError:
        raise ValueError(
            "%r is not a valid metric value. "
            "check neuralforecast library "
            "to get valid options." % name
        )
    return scorer


def smape(A, F):
    # src: https://stackoverflow.com/a/51440114
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def get_sklearn_metric(name: str) -> Callable[[NDArray, NDArray], NDArray | float] | None:
    import sklearn.metrics as sklm

    SCORER_MAPPING = {
        "mae": sklm.mean_absolute_error,
        "mape": sklm.mean_absolute_percentage_error,
        "mse": sklm.mean_squared_error,
        "msle": sklm.mean_squared_log_error,
        "rmse": functools.partial(sklm.mean_squared_error, squared=False),
        "smape": smape,
    }
    try:
        fun = SCORER_MAPPING[name]
        return fun
    except KeyError:
        log.warning(
            "%r is not a valid metric value. "
            "check sklearn.get_sco library "
            "to get valid options." % name
        )
        return None
