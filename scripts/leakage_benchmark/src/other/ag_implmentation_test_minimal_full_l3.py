import numpy as np
import openml
import pandas as pd

from autogluon.tabular import TabularPredictor


def get_data(tid: int, fold: int):
    # Get Task and dataset from OpenML and return split data
    oml_task = openml.tasks.get_task(tid, download_splits=True, download_data=True, download_qualities=False, download_features_meta_data=False)

    train_ind, test_ind = oml_task.get_train_test_split_indices(fold)
    X, *_ = oml_task.get_dataset().get_data(dataset_format="dataframe")

    return (
        X.iloc[train_ind, :].reset_index(drop=True),
        X.iloc[test_ind, :].reset_index(drop=True),
        oml_task.target_name,
        oml_task.task_type != "Supervised Classification",
    )


def _print_lb(leaderboard, task_id):
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard[leaderboard["model"].str.endswith("L1")].sort_values(by="score_val", ascending=True))
        print(leaderboard[~leaderboard["model"].str.endswith("L1")].sort_values(by="score_val", ascending=False))


def _run(task_id, metric, fold=0, print_res=True, enable_test=False):
    train_data, test_data, label, regression = get_data(task_id, fold)
    n_max_cols = 100
    n_max_train_instances = 500
    n_max_test_instances = 200

    # Sub sample instances
    train_data = train_data.sample(n=min(len(train_data), n_max_train_instances), random_state=0).reset_index(drop=True)
    test_data = test_data.sample(n=min(len(test_data), n_max_test_instances), random_state=0).reset_index(drop=True)

    # Sub sample columns
    cols = list(train_data.columns)
    cols.remove(label)
    if len(cols) > n_max_cols:
        cols = list(np.random.RandomState(42).choice(cols, replace=False, size=n_max_cols))
    train_data = train_data[cols + [label]]
    test_data = test_data[cols + [label]]

    # Run AutoGluon
    print(f"### task {task_id} and fold {fold} and shape {(train_data.shape, test_data.shape)}.")
    predictor = TabularPredictor(eval_metric=metric, label=label, verbosity=0)
    predictor.fit(
        train_data=train_data,
        num_gpus=0,
        num_bag_sets=1,
        num_stack_levels=1,
        num_bag_folds=6,
        hyperparameters={
                "FASTAI": [{}],
                "NN_TORCH": [{}],
                "RF": [{}],
                "XT": [{}],
                "GBM": [{}],
        },
        fit_weighted_ensemble=True,
        fit_full_last_level_weighted_ensemble=enable_test,
        full_weighted_ensemble_additionally=True,
        presets="best_quality",
        # time_limit=60
    )
    leaderboard = predictor.leaderboard(test_data, silent=True)[["model", "score_test", "score_val"]].sort_values(by="model").reset_index(drop=True)

    if print_res:
        _print_lb(leaderboard, task_id)

    return leaderboard


if __name__ == "__main__":
    _run(359938, "mse", enable_test=True)
    _run(359938, "mse")

    _run(359955, "roc_auc", enable_test=True)
    _run(359955, "roc_auc")

    _run(146217, "log_loss", enable_test=True)
    _run(146217, "log_loss")



