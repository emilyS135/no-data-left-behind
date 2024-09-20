import matplotlib

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

import seaborn as sns


matplotlib.use("Agg")
plt.style.use(["science", "no-latex"])


def catplot_by_station_and_measure(df):
    df.columns = [*df.columns[:-1], "result"]

    pattern = r"(.*)_((?:personnel_count|minute_capacity))"

    df[["ward", "measure"]] = df["unique_id"].str.extract(pattern, expand=True)
    df = df.drop(columns=["unique_id"])
    g = sns.catplot(
        y="ward",
        x="result",
        col="metric",
        row="measure",
        data=df,
        kind="bar",
        sharex=False,
        sharey=False,
        log=False,
    )
    return g
