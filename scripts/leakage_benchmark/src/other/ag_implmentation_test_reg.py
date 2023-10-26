import numpy as np
import openml
import pandas as pd
import sklearn

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
    # print("### Results for ", task_id)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard[leaderboard["model"].str.endswith("L1")].sort_values(by="score_val", ascending=True))
        print(leaderboard[~leaderboard["model"].str.endswith("L1")].sort_values(by="score_val", ascending=False))


def _run(task_id, metric, fold=0, print_res=True):
    l2_train_data, l2_test_data, label, regression = get_data(task_id, fold)
    n_max_cols = 100
    n_max_train_instances = 10000
    n_max_test_instances = 2000

    # Sub sample instances
    l2_train_data = l2_train_data.sample(n=min(len(l2_train_data), n_max_train_instances), random_state=0).reset_index(drop=True)
    l2_test_data = l2_test_data.sample(n=min(len(l2_test_data), n_max_test_instances), random_state=0).reset_index(drop=True)

    # Sub sample columns
    cols = list(l2_train_data.columns)
    cols.remove(label)
    if len(cols) > n_max_cols:
        cols = list(np.random.RandomState(42).choice(cols, replace=False, size=n_max_cols))
    l2_train_data = l2_train_data[cols + [label]]
    l2_test_data = l2_test_data[cols + [label]]

    # Run AutoGluon
    print(f"### task {task_id} and fold {fold} and shape {(l2_train_data.shape, l2_test_data.shape)}.")
    predictor = TabularPredictor(eval_metric=metric, label=label, verbosity=2)
    predictor.fit(
        train_data=l2_train_data,
        hyperparameters={
            "FASTAI": [{}, {"clipping": False, "ag_args": {"name_suffix": "_no_clipping"}}, {"clipping": False, "y_scaler": sklearn.preprocessing.MinMaxScaler(), "ag_args": {"name_suffix": "_no_clipping_custom_scaler"}}],
        },
        num_stack_levels=1,
        num_bag_sets=5,
        num_bag_folds=8,
        fit_weighted_ensemble=False,
    )
    leaderboard = predictor.leaderboard(l2_test_data, silent=True)[["model", "score_test", "score_val"]].sort_values(by="model").reset_index(drop=True)

    if print_res:
        _print_lb(leaderboard, task_id)

    return leaderboard


if __name__ == "__main__":
    _run(359938, "rmse", fold=3)  # leak minimal
