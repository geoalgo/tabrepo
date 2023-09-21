import json
import pickle
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def _score_spotter(leak_y, spotter_y):

    print("Balanced Accuracy:", balanced_accuracy_score(leak_y, spotter_y))
    cm = pd.DataFrame(confusion_matrix(leak_y, spotter_y), columns=['Spotter - False', 'Spotter - True'],
                      index=['GT - False', 'GT - True'])
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(cm)


def _get_metadata(tids):
    print("Getting MetaData")

    md_file_path = Path("./md_file.json")
    md_file = None
    if md_file_path.exists():
        with open(md_file_path, "r") as f:
            md_file = json.load(f)
        if set(md_file.keys()) == set(tids):
            return md_file
        else:
            md_file = None

    if md_file is None:
        md_file = {
            int(tid): openml.datasets.get_dataset(
                openml.tasks.get_task(
                    int(tid), download_splits=False, download_data=False, download_qualities=True, download_features_meta_data=False
                ).dataset_id,
                download_data=False,
                download_qualities=True,
                download_features_meta_data=False,
            ).qualities
            for tid in tids
        }
        with open(md_file_path, "w") as f:
            json.dump(md_file, f)

    return md_file

def _spot(l1_val, l2_val, l1_repo, l2_repo):
    leak_spotted = l1_val < l2_val
    leak_spotted = leak_spotted & (
        (l2_repo < l1_repo)
    )

    return leak_spotted

def _inspect_spotter(in_df, spotter_name):
    print(f"#### For Spotter: {spotter_name}")
    print("\n## Overall")
    _score_spotter(in_df['leak'], in_df['leak_spotted'])

    print("\n## Leak Is Possible At All")
    _score_spotter(in_df.loc[in_df["stacking_has_no_impact"] == False, 'leak'],
                   in_df.loc[in_df["stacking_has_no_impact"] == False, 'leak_spotted'])

    print("\n## Stacking Is Better")
    _score_spotter(in_df.loc[in_df["stacking_is_better"] == True, 'leak'],
                   in_df.loc[in_df["stacking_is_better"] == True, 'leak_spotted'])


with open(f'results_full_alt_metric.pkl', 'rb') as f:
    res_dict = pickle.load(f)
df = pd.DataFrame(res_dict)
md = _get_metadata(list(df['task_id']))
md = pd.DataFrame(md).T.dropna(axis=1)
df = df.merge(md, left_on='task_id', right_index=True, validate='1:1')
df.to_csv("./res.csv")

print(list(df[(df["leak"] == True) | df["leak_spotted"] == True]['task_id']))

# -- Current spotter
df['leak_spotted'] = _spot(df['l1_val'], df['l2_val'], df["am_l2_true_repo"], df['am_l2_repo'])
_inspect_spotter(df, "current_spotter")

# exit()
# df['leak_spotted'] = _spot_tests(df['l1_val'], df['l2_val'], df['l1_repo'], df['l2_repo'])
# _inspect_spotter(df, "test_spotter")

# -- Try to learn this
# df = df.loc[df["stacking_has_no_impact"] == False]
y = df["leak"]
X = df.drop(columns=["leak", "leak_spotted", "stacking_is_better", "stacking_has_no_impact", "task_id", "l1_test",
                     "l2_test"])
X = X[["am_l1_repo", "am_l2_true_repo"]]

from pysr import PySRRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
# - SKlearn
from sklearn.model_selection import LeaveOneOut, cross_val_predict

rng = 42 # np.random.RandomState(42)
oof_pred = cross_val_predict(RandomForestClassifier(random_state=rng, verbose=False), X, y,
                              cv=LeaveOneOut())
df['leak_spotted'] = oof_pred
_inspect_spotter(df, "model_fit")

# exit()

print('start')
model = PySRRegressor(
    niterations=500,
    binary_operators=["greater", "logical_and", "logical_or"],
    unary_operators=["neg"],
    complexity_of_constants=100000000000000000,
    maxsize=20,
    populations=30,
    annealing=True,
    verbosity=0,
)
# print(np.mean(cross_val_score(model, X, y,
#                               cv=LeaveOneOut(), scoring="balanced_accuracy")))

model.fit(X, y)
print(model.equations_.iloc[-1]['equation'])
y_pred = model.predict(X).round(0)
df['leak_spotted'] = y_pred
_inspect_spotter(df, "model_fit")