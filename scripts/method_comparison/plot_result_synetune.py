from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments import load_experiments_df

from tabrepo.loaders import Paths

def plot_results(df, title: str = None, colors: Dict = None, show_seeds: bool = False):
    method_col = "scheduler_name"
    if colors is None:
        cmap = cm.get_cmap("Set1")
        colors = {
            k: cmap(i)
            for i, k in enumerate(df[method_col].unique())
        }
    metric = df.loc[:, "metric_names"].values[0]
    mode = df.loc[:, "metric_mode"].values[0]

    fig, ax = plt.subplots()

    for algorithm in sorted(df[method_col].unique()):
        ts = []
        ys = []
        y_tests = []

        df_scheduler = df[df[method_col] == algorithm]
        for i, tuner_name in enumerate(df_scheduler.tuner_name.unique()):
            sub_df = df_scheduler[df_scheduler.tuner_name == tuner_name]
            sub_df = sub_df.sort_values(ST_TUNER_TIME)
            t = sub_df.loc[:, ST_TUNER_TIME].values
            y_best = (
                sub_df.loc[:, metric].cummax().values
                if mode == "max"
                else sub_df.loc[:, metric].cummin().values
            )
            y_test = sub_df.loc[:, "test_error"]
            if show_seeds and i == 0:
                print(y_best)
                ax.plot(t, y_best, color=colors[algorithm], alpha=0.2)
                ax.plot(t, y_test, color=colors[algorithm], alpha=0.2)

            ts.append(t)
            ys.append(y_best)
            y_tests.append(y_test)

        # compute the mean/std over time-series of different seeds at regular time-steps
        # start/stop at respectively first/last point available for all seeds
        t_min = max(tt[0] for tt in ts)
        t_max = min(tt[-1] for tt in ts)
        if t_min > t_max:
            continue
        t_range = np.linspace(t_min, t_max)

        # find the best value at each regularly spaced time-step from t_range
        y_ranges = []
        # y_ranges_tests = []

        for t, y, y_yesy in zip(ts, ys, y_tests):
            indices = np.searchsorted(t, t_range, side="left")
            y_ranges.append(y[indices])
            # y_ranges_tests.append(y_test[indices])
        y_ranges = np.stack(y_ranges)
        # y_test_ranges = np.stack(y_ranges_tests)

        mean = y_ranges.mean(axis=0)
        std = y_ranges.std(axis=0)
        ax.fill_between(
            t_range,
            mean - std,
            mean + std,
            color=colors[algorithm],
            alpha=0.1,
        )
        ax.plot(t_range, mean, color=colors[algorithm], label=algorithm)

    ax.set_xlabel("wallclock time")
    ax.set_ylabel(metric)
    ax.legend()
    if title:
        ax.set_title(title)

    # (Path(__file__).parent / "figures").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(f"figures/{title}.png")
    plt.tight_layout()
    plt.show()


# if __name__ == '__main__':
expname = "sEcX3"
csv_filename = Paths.results_root / f"{expname}.csv"

name_filter = lambda path: expname in str(path)
df_tuning = load_experiments_df(path_filter=name_filter)
df_tuning["fold"] = df_tuning.apply(lambda row: 1 + int(row['tuner_name'].split("fold-")[1].split("-")[0]), axis=1)
df_train_error = df_tuning.groupby(["scheduler_name", "fold"]).mean()['train_error'].reset_index()
df_train_error = df_train_error.rename(columns={"scheduler_name": "searcher", "train_error": "error"})
df_train_error["split"] = "train"

df_results = pd.read_csv(csv_filename)
# for row in df_results.to_dict(orient="records"):
#     print(row)
df_test_error = df_results.rename(columns={"test-score": "error"})[["searcher", "fold", "error"]]
df_test_error["split"] = "test"
df_error = pd.concat([df_train_error, df_test_error], ignore_index=True)
df_error['searcher'] = df_error["searcher"].str.lower()

print(df_error.to_string())
print(df_error.pivot_table(values="error", columns="fold", index=['split', 'searcher']).to_string())

plot_results(df_tuning, show_seeds=False, title="Ensemble search")

