from abc import ABC, abstractmethod

import pandas as pd


class DataModule(ABC):
    @abstractmethod
    def pipeline(self, target: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_timeseries_data(self, target: str) -> pd.DataFrame:
        pass
