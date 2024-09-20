from datetime import timedelta
from pathlib import Path

import pandas as pd
from src.data.datamodule import DataModule
from src.data import staff_data as ltsf_staff
from src.utils import get_logger

log = get_logger(__name__)


class LTSFDataModule(DataModule):
    def __init__(
        self, path: str, file_name: str, force_rebuild: bool, freq: timedelta, target: str
    ) -> None:
        super().__init__()
        self.path = path
        self.file_name = file_name
        self.force_rebuild = force_rebuild
        self.freq = freq
        self.target = target
        self._data: pd.DataFrame | None = None
        self._futr_exog: pd.DataFrame | None = None
        self._hist_exog: pd.DataFrame | None = None
        self._stat_exog: pd.DataFrame | None = None

    def pipeline(
        self,
        feature_list: list[str] | None = None,
        target: str | None = None,
    ) -> pd.DataFrame:
        """Loads or generates the feature dataframe.

        Args:
            feature_list (list[str]): unused. Defaults to None.
            target (str): name of the target variable. Defaults to None.

        Raises:
            ValueError: if the dataframe can't be generated

        Returns:
            pd.DataFrame: feature set
        """
        if target is not None:
            # reset data
            self.target = target
            self._data = None
        return self.data

    def get_tabular_data(self, feature_list: list[str], target: str) -> pd.DataFrame:
        raise NotImplementedError("Data contains timeseries data only.")

    def get_windowed_data(self) -> pd.DataFrame:
        raise NotImplementedError("this is done automagically.")

    def get_raw_data(self) -> tuple[pd.DataFrame, ...]:
        # load nursing staff data
        ret = pd.read_csv(self.path + "/" + self.file_name, delimiter=";")
        if isinstance(ret, tuple):
            return ret
        else:
            return (ret,)

    def get_timeseries_data(self, target: str | None = None) -> pd.DataFrame:
        ret = self.get_raw_data()
        staff, *_ = ret
        base_path = Path(self.path)
        if target is not None:
            if target != self.target:
                # this is only for interface consistency
                log.warn(
                    f"new target column selected. was {self.target}, now building timeseries data for {target}"
                )
                self.target = target
        else:
            target = self.target

        # convert date column to datetime

        staff["date"] = pd.to_datetime(staff["date"], format="%d.%m.%Y %H:%M")

        capacity_ts, futr_df = ltsf_staff.build_all_interval_data(
            staff, base_path, pd.Timedelta(self.freq), save=True
        )
        # bring dataframes into neuralforecast specific format

        capacity_ts = capacity_ts.rename(columns={"date": "ds", target: "y"})

        # add target identifier column
        capacity_ts["unique_id"] = target
        futr_df["unique_id"] = target

        futr_df = futr_df.rename(columns={"date": "ds"})

        capacity_ts = capacity_ts.reset_index(drop=True)

        self._data = capacity_ts
        self._futr_exog = futr_df

        log.info("Finished building Datamodule.")
        log.info("care capacity timeseries head:")
        log.info(self._data.head().to_string())
        log.info("future exogenous timeseries head:")
        log.info(self._futr_exog.head().to_string())
        return self._data

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = self.get_timeseries_data()
        return self._data

    @property
    def futr_exog(self) -> pd.DataFrame | None:
        if self._futr_exog is None:
            self.get_timeseries_data()
        return self._futr_exog
