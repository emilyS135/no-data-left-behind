import hashlib
import logging
import os
from darts import TimeSeries
import requests
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np


def split_train_test(dataset: pd.DataFrame, ratio_or_start: float | str):
    assert_feature_existence(dataset, ["ds"])
    if isinstance(ratio_or_start, str):
        # assuming timestamp
        test_date = pd.Timestamp(ratio_or_start)
        print(f"debug: received test split start fixed date: {test_date}")
    else:
        n_usable_steps = dataset.ds.nunique()
        usable_step_idx = np.arange(n_usable_steps)
        test_size = round(n_usable_steps * ratio_or_start)
        test_start = np.sort(usable_step_idx)[-test_size]
        test_date = dataset.iloc[test_start]["ds"].floor("d")

    # no gap, no Lookback or Horizon consideration, NF should take care of all that.
    train = dataset[dataset.ds < test_date].copy()
    test = dataset[dataset.ds >= test_date].copy()
    return train, test


# src: tsai + sklearn
def get_walk_forward_splits(
    df,  # time series we need to split
    test_start_ts,
    horizon,
    step_size,
    n_splits: int | None = None,  # # of splits
    gap=0.0,  # # of samples to exclude from the end of each train set before the test set. Entered as an int or a float
    test_size: int | None = None,  # set either splits or size
):
    total_test_split = df[df.ds > test_start_ts]
    test_start_idx = df.loc[df.ds == test_start_ts].index
    n_samples = len(total_test_split)
    n_samples_total = len(df)

    if n_splits is None and test_size is None:
        raise Exception("you must define `n_splits` or `test_size`.")
    if test_size is None and n_splits is not None:
        test_size = n_samples // n_splits
        print(f"set test size to {test_size} based on n splits = {n_splits}")
    elif n_splits is None:
        _, rm = np.divmod(n_samples - horizon, step_size)
        if rm != 0:
            print(f"`test_size - h` should be module `step_size`, removing {rm} samples")
            test_size = test_size - rm
        n_splits = int((n_samples - horizon) / step_size) + 1
        print(f"set n splits = {n_splits} based on test size {test_size}")

    assert (test_size + gap) * n_splits < len(df), "reduce test_size,  gap or n_splits"
    if n_samples - gap - (test_size * n_splits) < 0:
        raise ValueError(
            f"Too many splits={n_splits} for number of samples"
            f"={n_samples} with test_size={test_size} and gap={gap}."
        )
    all_indices = np.arange(n_samples_total)
    test_starts = range(n_samples - n_splits * test_size, n_samples, test_size)
    for test_start in test_starts:
        train_end_abs = (test_start_idx + (test_start - gap))[0]  # int64index to int
        test_start_abs = (test_start_idx + test_start)[0]
        assert test_start_abs >= train_end_abs, "train ends after test start"
        yield (
            df.iloc[all_indices[:train_end_abs]],
            df.iloc[all_indices[test_start_abs : test_start_abs + test_size]],
        )


def find_last_modified_file(path: Path, pattern: str) -> Path:
    data_files = sorted(
        list(path.glob(pattern)),
        key=os.path.getmtime,
        reverse=True,
    )
    if len(data_files) > 0:
        return data_files[0]
    else:
        raise FileNotFoundError(f"no file found in {path} matching pattern {pattern}")



def assert_feature_existence(df: pd.DataFrame | pd.Series, col_names: list[str]):
    for col in col_names:
        if isinstance(df, pd.DataFrame):
            assert col in df.columns, f"Column '{col}' not found. Given: {df.columns}"
        elif isinstance(df, pd.Series):
            assert col in df.index, f"Series Index '{col}' not found. Given: {df.index}"
        else:
            raise TypeError(f"Received '{type(df)}'. Expected pandas DataFrame or Series.")


def save_processed_dataframe(
    df: pd.DataFrame,
    base_path: Path,
    name: str,
    config: dict | None = None,
    add_file_ending: bool = False,
    subdir: str | None = None,
):
    processed_dir = base_path.parent.parent / "processed"
    if subdir is not None:
        processed_dir = processed_dir / subdir

    filename = name
    # Add additional identifiers to the file ending
    if add_file_ending:
        length = len(df) // 1_000_000
        # get a unique hash reflecting the configuration values
        confighash = hashlib.md5((str(config).encode())).hexdigest()
        filename = filename + f"_{length}M_{confighash}"

    processed_dir.mkdir(parents=True, exist_ok=True)
    save_feather(df, processed_dir, filename)


def save_yaml(data: dict, base_path: Path, name: str, subdir: str | None = None):
    import yaml

    if subdir is not None:
        out_dir = base_path / subdir
    else:
        out_dir = base_path
    out_dir.mkdir(parents=True, exist_ok=True)
    curr_date = pd.Timestamp.now().strftime("%Y%m%d")
    new_filename = (out_dir / f"{name}_{curr_date}").with_suffix(".yaml")
    with open(new_filename, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    logging.info(f"Saved to {new_filename}")


def save_csv(data: pd.DataFrame, base_path: Path, name: str, subdir: str | None = None):
    if subdir is not None:
        out_dir = base_path / subdir
    else:
        out_dir = base_path
    out_dir.mkdir(parents=True, exist_ok=True)
    curr_date = pd.Timestamp.now().strftime("%Y%m%d")
    new_filename = (out_dir / f"{name}_{curr_date}").with_suffix(".csv")
    data.to_csv(new_filename)
    logging.info(f"Saved to {new_filename}")




def save_feather(df: pd.DataFrame, path: Path, name: str):
    curr_date = pd.Timestamp.now().strftime("%Y%m%d")
    new_filename = path / f"{name}_{curr_date}"

    # we do not need the custom index, and feather does not support it.
    df.reset_index(drop=True, inplace=True)
    df.to_feather(new_filename.with_suffix(".feather"))
    logging.info(f"Saved to {new_filename}.feather")



def download_external_data_ltsf(
    base_path: Path,
    names: List[str] | None = None,
) -> None:
    """Utility Function for downloading necessary external data. Can be used to download
    everything, or only specific files, specified in a List of names.

    Args:
        base_path (Path): Path were the external files will be placed
        names (List[str] | None): List of the names of files to download. Viable options are:
                    'public_holidays'
                    'school_holidays'
                    'influenza'
                    Throws Key Error if names is specified but contains wrong keys.

    Returns:
        None
    """

    urls = {
        "public_holidays": "https://openholidaysapi.org/PublicHolidays?countryIsoCode=DE&languageIsoCode=DE&validFrom=2021-06-01&validTo=2024-05-30&subdivisionCode=DE-BY",
        "school_holidays": "https://openholidaysapi.org/SchoolHolidays?countryIsoCode=DE&languageIsoCode=DE&validFrom=2021-06-01&validTo=2024-05-30&subdivisionCode=DE-BY",
        "influenza": "https://raw.githubusercontent.com/robert-koch-institut/Influenzafaelle_in_Deutschland/2024-04-04/IfSG_Influenzafaelle.tsv",
        "are/ili": "https://raw.githubusercontent.com/robert-koch-institut/GrippeWeb_Daten_des_Wochenberichts/2024-04-25/GrippeWeb_Daten_des_Wochenberichts.tsv",
    }
    filenames = {
        "public_holidays": "public_holidays_1.json",
        "school_holidays": "school_holidays_1.json",
        "influenza": "IfSG_Influenzafaelle_1.tsv",
        "are/ili": "GrippeWeb_Daten_des_Wochenberichts_1.tsv",
    }

    keys = urls.keys()

    for name in names:
        if name not in keys:
            raise KeyError(
                f"{name} is not a valid key for an external file! Only {keys} are Options"
            )

    if names is not None:
        urls, filenames = [urls.get(key) for key in names], [filenames.get(key) for key in names]
    else:
        urls, filenames = (
            [urls.get(key) for key in urls.keys()],
            [filenames.get(key) for key in filenames.keys()],
        )

    for url, filename in zip(urls, filenames):
        with requests.get(url, stream=True) as response:
            with open(os.path.join(base_path, filename), mode="wb") as file:
                for chunk in response.iter_content(chunk_size=10 * 1024):
                    file.write(chunk)


def darts_series_from_df(
    dataframe: pd.DataFrame,
    freq: str,
    hist_exog_list: list[str] | None = None,
    futr_exog_list: list[str] | None = None,
) -> tuple[TimeSeries, TimeSeries | None, TimeSeries | None]:
    """
    Create TimeSeries objects from a pandas DataFrame for use in Darts.

    Args:
    - dataframe: The input DataFrame containing the time series data.
    - freq: The frequency of the time series data (e.g. 'D' for daily, 'M' for monthly, etc.).
    - hist_exog_list: A list of column names in the DataFrame that contain historical exogenous variables.
    - futr_exog_list: A list of column names in the DataFrame that contain future exogenous variables.

    Returns:
    - A tuple of three TimeSeries objects:
      1. The main time series.
      2. The historical exogenous variables (or None if no historical exogenous variables are provided).
      3. The future exogenous variables (or None if no future exogenous variables are provided).
    """
    series = TimeSeries.from_dataframe(dataframe, time_col="ds", value_cols="y", freq=freq)
    if hist_exog_list:
        past_covariates = TimeSeries.from_dataframe(
            dataframe,
            time_col="ds",
            value_cols=hist_exog_list,
            freq=freq,
        )
    else:
        past_covariates = None
    if futr_exog_list:
        future_covariates = TimeSeries.from_dataframe(
            dataframe,
            time_col="ds",
            value_cols=futr_exog_list,
            freq=freq,
        )
    else:
        future_covariates = None
    return series, past_covariates, future_covariates


def inv_transform(scaler, data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Inverse transform a scaled pandas DataFrame column using a StandardScaler.

    Args:
    scaler (StandardScaler): The scaler used to transform the data initially.
    data (pd.DataFrame): The DataFrame containing the column to be inverse transformed.
    col_name (str): The name of the column to be inverse transformed.

    Returns:
    pd.DataFrame: A DataFrame with the inverse transformed column.

    """
    import numpy as np

    dummy = pd.DataFrame(
        np.zeros((len(data), len(scaler.feature_names_in_))),
        columns=scaler.feature_names_in_,
        index=data.index,
    )
    dummy[col_name] = data.values
    dummy = pd.DataFrame(
        scaler.inverse_transform(dummy), columns=scaler.feature_names_in_, index=data.index
    )
    return dummy[[col_name]]
