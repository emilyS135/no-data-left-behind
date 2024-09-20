import json
from datetime import datetime

import numpy as np
from matplotlib.figure import Figure

from src.logging.mlflow import MlflowLogger


def flatten_dict(d, parent_key="", sep=" - "):
    flat_dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, new_key, sep=sep))
        else:
            flat_dict[new_key] = v
    return flat_dict


def save_dict_of_nparrays_to_json(
    data: dict[str, np.array], dir: str, name_tag: str, logger: MlflowLogger | None
) -> None:
    # Convert the NumPy arrays to lists of lists
    serializable_data = {k: v.tolist() for k, v in data.items()}

    # Get the current time and format it as a string suitable for a filename
    # Example format: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{dir}/{name_tag}_{timestamp}.json"

    # Write data to JSON file with the timestamped filename
    with open(filename, "w") as json_file:
        json.dump(serializable_data, json_file, indent=4)

    if logger:
        logger.log_file(filename)


def save_dict_of_list_to_json(
    data: dict[str, np.array], dir: str, name_tag: str, logger: MlflowLogger | None
) -> None:
    # Get the current time and format it as a string suitable for a filename
    serializable_data = {k: np.array(v).astype(float).tolist() for k, v in data.items()}
    # Example format: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{dir}/{name_tag}_{timestamp}.json"

    # Write data to JSON file with the timestamped filename
    with open(filename, "w") as json_file:
        json.dump(serializable_data, json_file, indent=4)
    if logger:
        logger.log_file(filename)


def save_dict_of_dict_of_list_to_json(
    data: dict[str, dict[str, np.array]], dir: str, name_tag: str, logger: MlflowLogger | None
) -> None:
    # Get the current time and format it as a string suitable for a filename
    serializable_data = {
        k: {k1: np.array(v1).astype(float).tolist() for k1, v1 in v.items()}
        for k, v in data.items()
    }
    # Example format: YYYYMMDD_HHMMSS
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{dir}/{name_tag}.json"

    # Write data to JSON file with the timestamped filename
    with open(filename, "w") as json_file:
        json.dump(serializable_data, json_file, indent=4)
    if logger:
        logger.log_file(filename)


def save_dict_of_dict_of_dict_of_list_to_json(
    data: dict[str, dict[str, dict[str, np.array]]],
    dir: str,
    name_tag: str,
    logger: MlflowLogger | None,
):
    # Get the current time and format it as a string suitable for a filename
    serializable_data = {
        k: {
            k1: {k2: np.array(v2).astype(float).tolist() for k2, v2 in v1.items()}
            for k1, v1 in v.items()
        }
        for k, v in data.items()
    }
    # Example format: YYYYMMDD_HHMMSS
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{dir}/{name_tag}.json"

    # Write data to JSON file with the timestamped filename
    with open(filename, "w") as json_file:
        json.dump(serializable_data, json_file, indent=4)
    if logger:
        logger.log_file(filename)


def save_dict_to_json(
    data: dict[str, np.array], dir: str, name_tag: str, logger: MlflowLogger | None
) -> None:
    # Convert the NumPy arrays to lists of lists
    serializable_data = {k: v.tolist() for k, v in data.items()}

    # Get the current time and format it as a string suitable for a filename
    # Example format: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{dir}/{name_tag}_{timestamp}.json"

    # Write data to JSON file with the timestamped filename
    with open(filename, "w") as json_file:
        json.dump(serializable_data, json_file, indent=4)

    if logger:
        logger.log_file(filename)


def save_figure_to_image(figure: Figure, directory: str, name_tag: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{directory}/{name_tag}_{timestamp}.pdf"

    figure.savefig(filename)
