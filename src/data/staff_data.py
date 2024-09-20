import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.utils import (
    find_last_modified_file,
    save_processed_dataframe,
    download_external_data_ltsf,
)

log = logging.getLogger(__name__)


def build_all_interval_data(
    interval_data: pd.DataFrame,
    base_path: Path,
    freq: pd.Timedelta,
    save: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds a dataframe of data per datetime in intervals.

    Args:
        base_path (Path): base path to the directory of the raw data
        freq (float): the frequency of datetimes to generate data rows for
        long_format (bool): return the data in wide (false) or long (true) format

    Returns:
        pd.DataFrame: data keyed by datetimes in intervals, including:
            - Staff Capacity in personnel count
            - if there are holidays on the datetime
            - the distance to each holiday
            - cyclical encoded calendar features
            - influenza occurrences recorded during this datetime
    """

    for func in [add_holiday_data, add_incidence_data, add_calendar_data]:
        interval_data = func(interval_data, base_path)
    interval_data["public_holiday"] = interval_data["public_holiday"].astype("int")
    interval_data["school_holiday"] = interval_data["school_holiday"].astype("int")

    futr_col_substrings = [
        "date",
        "public_holiday",
        "school_holiday",
        "_planned_absences",
        "sin_",
        "cos_",
        "distance_to",
    ]
    futr_col_names = [
        col for col in interval_data.columns if any(s in col for s in futr_col_substrings)
    ]

    futr_df = interval_data[futr_col_names].copy()

    if save:
        save_processed_dataframe(interval_data, base_path, "exog_interval_staff_capacity")

    return interval_data, futr_df


def sinus_transformation(period, x):
    return np.sin(x / period * 2 * np.pi)


def cosinus_transformation(period, x):
    return np.cos(x / period * 2 * np.pi)


def add_calendar_data(
    df: pd.DataFrame,
    base_path: Path,
    date_column: str = "date",
) -> pd.DataFrame:
    """Adds calendar features in cyclic sinus and cosine encoding to the dataframe
    added features: day of week, day of year, day of month, month of year, hour of day

    Args:
    df (DataFrame): the dataframe to add the holiday flags into
    base_path (Path): base path to the directory of the holidays raw data
    date_column (str): the datetime column of the dataframe to use


    Returns:
        pd.DataFrame: copy of the provided dataframe with the added calendar features
    """
    log.info("adding calendar features...")

    # day of week
    df["sin_day_of_week"] = df["date"].apply(lambda x: x.day_of_week)
    df["cos_day_of_week"] = df["date"].apply(lambda x: cosinus_transformation(7, x.day_of_week))

    # day of year
    df["sin_day_of_year"] = df["date"].apply(lambda x: sinus_transformation(365, x.day_of_year))
    df["cos_day_of_year"] = df["date"].apply(lambda x: cosinus_transformation(365, x.day_of_year))

    # day of month
    df["sin_day_of_month"] = df["date"].apply(lambda x: sinus_transformation(31, x.day))
    df["cos_day_of_month"] = df["date"].apply(lambda x: cosinus_transformation(31, x.day))

    # month of year
    df["sin_month_of_year"] = df["date"].apply(lambda x: sinus_transformation(12, x.month))
    df["cos_month_of_year"] = df["date"].apply(lambda x: cosinus_transformation(12, x.month))

    # hour of day
    df["sin_hour_of_day"] = df["date"].apply(lambda x: sinus_transformation(24, x.hour))
    df["cos_hour_of_day"] = df["date"].apply(lambda x: cosinus_transformation(24, x.hour))

    return df


def add_holiday_data(
    df: pd.DataFrame,
    base_path: Path,
    date_column: str = "date",
) -> pd.DataFrame:
    """Adds to a dataframe with a datetime column a flag for each datetime,
    marking whether it lies inside a public or school holiday.
    The dataset containing the holidays is expected to be prefiltered for Bayern only.
    Retrieve the raw data as JSON from:
    https://openholidaysapi.org/PublicHolidays?countryIsoCode=DE&languageIsoCode=DE&validFrom=2021-06-01&validTo=2024-05-30&subdivisionCode=DE-BY
    https://openholidaysapi.org/SchoolHolidays?countryIsoCode=DE&languageIsoCode=DE&validFrom=2021-06-01&validTo=2024-05-30&subdivisionCode=DE-BY

    Args:
        df (DataFrame): the dataframe to add the holiday flags into
        base_path (Path): base path to the directory of the holidays raw data
        date_column (str): the datetime column of the dataframe to use

    Returns:
        pd.DataFrame: copy of the provided dataframe with the added holiday flags
    """
    log.info("adding distance to holiday features...")

    # Get data file as one dataframe
    try:
        public_holidays_file = find_last_modified_file(base_path, "public_holidays_*.json")
        log.info(f"found public holidays file: {public_holidays_file}")
    except FileNotFoundError:
        log.info("File for public holidays was not found and will be downloaded!")
        download_external_data_ltsf(base_path=base_path, names=["public_holidays"])
        public_holidays_file = find_last_modified_file(base_path, "public_holidays_*.json")

    try:
        school_holidays_file = find_last_modified_file(base_path, "school_holidays_*.json")
        log.info(f"found school holidays file: {school_holidays_file}")
    except FileNotFoundError:
        log.info("File for school holidays was not found and will be downloaded!")
        download_external_data_ltsf(base_path=base_path, names=["school_holidays"])
        school_holidays_file = find_last_modified_file(base_path, "school_holidays_*.json")

    all_holidays = pd.concat(
        [pd.read_json(public_holidays_file), pd.read_json(school_holidays_file)]
    )

    # Convert date strings to datetimes
    # and end date should include the whole day itself
    all_holidays[["startDate", "endDate"]] = all_holidays[["startDate", "endDate"]].apply(
        pd.to_datetime
    )
    all_holidays["endDate"] += timedelta(days=1) - timedelta(microseconds=1)

    def normalize_text(s: str):
        return s.translate(
            str.maketrans({" ": "_", "ä": "ae", "ö": "oe", "ü": "ue", ".": "", "-": "", "ß": "ss"})
        )

    all_holidays["name"] = all_holidays["name"].apply(
        lambda x: normalize_text(x[0]["text"].lower())
    )

    # For all datetimes, mark if they fall into a holiday, and if not how far they are from the holidays.
    df_appended = df.assign(public_holiday=False, school_holiday=False)

    names = list(all_holidays["name"].apply(lambda x: "distance_to_" + x))
    all_holidays = all_holidays.reset_index(drop=True)
    df_appended[names] = pd.DataFrame(
        [[int(pd.Timedelta(0).total_seconds()) for _ in names]], index=df.index
    )

    all_holidays = all_holidays.groupby("name").agg(
        {
            **{col: list for col in ["startDate", "endDate"]},
            **{
                col: "first" for col in all_holidays.columns.drop(["name", "startDate", "endDate"])
            },
        }
    )
    all_holidays = all_holidays.reset_index()

    for _, row in all_holidays.iterrows():
        temp = []
        for idx in range(len(row["startDate"])):
            temp.append(
                (df_appended[date_column] >= row["startDate"][idx])
                & (df_appended[date_column] <= row["endDate"][idx])
            )
        in_holiday = np.array([any(t) for t in zip(*temp)])

        target = "public_holiday" if row["type"] == "Public" else "school_holiday"
        df_appended.loc[in_holiday, target] = True
        d2p = distance_to_period(
            df_appended.loc[~in_holiday, date_column], row[["startDate", "endDate"]]
        )
        df_appended.loc[~in_holiday, "distance_to_" + row["name"]] = d2p

    return df_appended


def distance_to_period(days: pd.Series, holidays: pd.DataFrame):
    temp = []
    for idx in range(len(holidays["startDate"])):
        before = abs(holidays["startDate"][idx] - days)
        after = abs(holidays["endDate"][idx] - days)
        result = []
        for id, aft in enumerate(after):
            result.append(int((before.iloc[id] if before.iloc[id] < aft else aft).days) + 1)
        result = np.array(result)
        temp.append(result)

    distances = np.array([min(t) for t in zip(*temp)])
    return distances


def add_incidence_data(
    df: pd.DataFrame | None,
    base_path: Path,
    date_column: str = "date",
) -> pd.DataFrame:
    """Adds to a dataframe with a datetime column the regional incidence for several illnesses each date:
        - influenza
        - Akutute Atemwegserkrankung (ARE)
        - Influenza like illnesses (ILI)
    Retreive the raw data from:
    https://github.com/robert-koch-institut/Influenzafaelle_in_Deutschland/blob/2024-04-25/IfSG_Influenzafaelle.tsv
    https://raw.githubusercontent.com/robert-koch-institut/Influenzafaelle_in_Deutschland/2024-04-25/IfSG_Influenzafaelle.tsv

    https://github.com/robert-koch-institut/GrippeWeb_Daten_des_Wochenberichts/blob/2024-04-25/GrippeWeb_Daten_des_Wochenberichts.tsv
    https://raw.githubusercontent.com/robert-koch-institut/GrippeWeb_Daten_des_Wochenberichts/2024-04-25/GrippeWeb_Daten_des_Wochenberichts.tsv

    Args:
        df (DataFrame | None): the dataframe to add the incidences into
        base_path (Path): base path to the directory of the incidence raw data
        date_column (str): the datetime column of the dataframe to use

    Returns:
        pd.Dataframe: copy of the provided dataframe with the added incidence values or new dataframe

    Licence:
        -> licences/LICENSE_RKI_Attribution_4_0_International
    """
    all_data = pd.DataFrame(columns=["Meldewoche", "Erkrankung", "Inzidenz"])

    # Get data file as dataframe
    try:
        data_file = find_last_modified_file(base_path, "IfSG_Influenzafaelle*.tsv")
        log.info(f"found influenza occurrences file: {data_file}")
    except FileNotFoundError:
        log.info("File for influenza occurrences was not found and will be downloaded!")
        download_external_data_ltsf(base_path=base_path, names=["influenza"])
        data_file = find_last_modified_file(base_path, "IfSG_Influenzafaelle*.tsv")

    # Get only data for the desired region, and sum the incidence per week.
    influenza_data = (
        pd.read_table(data_file)
        .query("Region == 'Bayern' and Altersgruppe == '00+'")
        .assign(Erkrankung="influenza")
    )
    all_data = pd.concat([all_data, influenza_data[["Meldewoche", "Erkrankung", "Inzidenz"]]])

    # ARE / ILI data
    try:
        data_file = find_last_modified_file(base_path, "GrippeWeb_Daten_des_Wochenberichts*.tsv")
        log.info(f"found ARE/ILI occurences file: {data_file}")
    except FileNotFoundError:
        log.info("File for ARE/ILI occurrences was not found and will be downloaded!")
        download_external_data_ltsf(base_path=base_path, names=["are/ili"])
        data_file = find_last_modified_file(base_path, "GrippeWeb_Daten_des_Wochenberichts*.tsv")

    are_ili_data = (
        pd.read_table(data_file)
        .query("Region == 'Sueden' and Altersgruppe == '00+'")
        .rename(columns={"Kalenderwoche": "Meldewoche"})
    )
    all_data = pd.concat([all_data, are_ili_data[["Meldewoche", "Erkrankung", "Inzidenz"]]])

    # Add start and end dates based on the elicitation week.
    def _week_of_year_as_time_range(year_week_string) -> tuple[datetime, datetime]:
        """Returns datetime for the start and end of the week by a given [year]-[week] string.

        week string format: '2020-W01' for first week of 2020
        """
        start_date = datetime.strptime(year_week_string + "-1", "%G-W%V-%u")
        end_date = start_date + timedelta(days=7) - timedelta(microseconds=1)
        return (start_date, end_date)

    all_data[["startDate", "endDate"]] = [
        (_week_of_year_as_time_range(x)) for x in all_data["Meldewoche"]
    ]

    # Add columns for all illnesses,
    # assign all time intervalls of the dataframe the according incidences.
    df_appended = df.copy()
    for illness in all_data["Erkrankung"].unique():
        df_appended[f"{illness}_incidence"] = 0
    for _, row in all_data.iterrows():
        intervall_in_occurence = (df_appended[date_column] >= row["startDate"]) & (
            df_appended[date_column] <= row["endDate"]
        )
        df_appended.loc[intervall_in_occurence, f"{row['Erkrankung']}_incidence"] = row["Inzidenz"]

    return df_appended
