
import numpy as np

from src.utils import get_logger
from src.utils.types import MetricCallable

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


