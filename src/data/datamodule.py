from abc import ABC, abstractmethod

import pandas as pd


class DataModule(ABC):
    @abstractmethod
    def pipeline(self, data_type: str, feature_list: list[str], target: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_tabular_data(self, feature_list: list[str], target: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_windowed_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_timeseries_data(self, target: str) -> pd.DataFrame:
        pass
