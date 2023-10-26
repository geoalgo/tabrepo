from shutil import rmtree

import numpy as np
import openml
import pandas as pd
from scipy.special import xlogy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from autogluon.core.metrics import get_metric
from autogluon.tabular import TabularDataset, TabularPredictor


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
    print("### Results for ", task_id)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard.sort_values(by="score_val", ascending=False))


def _run(task_id, metric, test=True, fold=0, print_res=True):
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

    print(l2_train_data.shape, l2_test_data.shape)

    # Run AutoGluon
    print(f"Start running AutoGluon on L2 data for task {task_id} and fold {fold}.")
    predictor = TabularPredictor(eval_metric=metric, label=label, verbosity=0, learner_kwargs=dict(random_state=0))
    predictor.fit(
        train_data=l2_train_data,
        hyperparameters={
            1: {
                "FASTAI": [{}],
                "NN_TORCH": [{}],
                "RF": [{}],
                "XT": [{}],
                "GBM": [{}],
            },
            2: {
                "RF": [{}],
                "GBM": [
                    # {},
                    {
                        "ir": True,
                        "ag_args": {"name_suffix": "_ir"},
                    },
                    {
                        "ir": True,
                        "ir_fit_full": True,
                        "ag_args": {"name_suffix": "_ir_fit_full"},
                    },
                    {
                        "ir": True, "ir_bounds": False,
                        "ag_args": {"name_suffix": "_ir_bounds"},
                    },
                    {
                        "ir": True,  "ir_bounds": False,
                        "ir_fit_full": True,
                        "ag_args": {"name_suffix": "_ir_fit_full_bounds"},
                    },
                    # {
                    #     "ir": True,
                    #     "ir_p": True,
                    #     "ir_p_val": "full",
                    #     "ag_args": {"name_suffix": "_ir_p_vp-full"},
                    # },
                    # {
                    #     "ir": True,
                    #     "ir_p": True,
                    #     "ir_p_val": "re",
                    #     "ag_args": {"name_suffix": "_ir_p_vp-re"},
                    # },
                ],
            },
        },
        num_stack_levels=1,
        num_bag_sets=1,
        num_bag_folds=6,
        fit_weighted_ensemble=True,
        # presets='best_quality',
        # time_limit=60,
        # ag_args_fit=dict(ag_args_fit_key=True),
        # ag_args=dict(ag_args_key=True),
        ag_args_ensemble=dict(
            fold_fitting_strategy="sequential_local",
            # ag_args_fit_key=True
            # nested=True,
            # nested_num_folds=8,
        ),
    )
    leaderboard = predictor.leaderboard(l2_test_data, silent=True)[["model", "score_test", "score_val"]].sort_values(by="model").reset_index(drop=True)

    if print_res:
        _print_lb(leaderboard, task_id)

    rmtree(predictor.path)

    return leaderboard


def _run_for_all_folds(task_id, metric, test=True):
    lb_list = []
    for fold_i in [0, 2, 5, 1]:  # range(10):
        lb = _run(task_id, metric, test=test, fold=fold_i, print_res=True)
        lb_list.append(lb)
    f_lb = pd.concat(lb_list).groupby("model").mean()
    _print_lb(f_lb, task_id)


if __name__ == "__main__":
    # --- Other
    # _run(359974, "log_loss")  # 359964
    # _run(359974, "log_loss", False)

    # _run(146217, "log_loss")
    # _run(146217, "log_loss", False)

    # -- Binary
    # _run(359955, "roc_auc", fold=1)

    # _run(359955, "roc_auc", fold=2)

    # blood-transfusion-service-center
    _run_for_all_folds(359955, "roc_auc")  # leak minimal
    # Across 10 folds
    #                      score_test  score_val
    # model
    # WeightedEnsemble_L3    0.702775   0.744087
    # LightGBM_BAG_L2        0.712636   0.733274
    # RandomForest_BAG_L2    0.671829   0.723778
    # WeightedEnsemble_L2    0.733964   0.722367
    # LightGBM_BAG_L1        0.742082   0.721205
    # ExtraTrees_BAG_L2      0.690296   0.719231
    # ExtraTrees_BAG_L1      0.686891   0.681342
    # RandomForest_BAG_L1    0.691047   0.674218

    # _run_for_all_folds(359955, "roc_auc", False )  # leak minimal
    # Across 10 folds
    #                      score_test  score_val
    # model
    # WeightedEnsemble_L3    0.698303   0.855896
    # ExtraTrees_BAG_L2      0.702336   0.855482
    # RandomForest_BAG_L2    0.682448   0.838359
    # LightGBM_BAG_L2        0.664993   0.786711
    # WeightedEnsemble_L2    0.733964   0.722367
    # LightGBM_BAG_L1        0.742082   0.721205
    # ExtraTrees_BAG_L1      0.686891   0.681342
    # RandomForest_BAG_L1    0.691047   0.674218

    # Titanic
    # _run(361339, "roc_auc")  # leak minimal
    # _run(361339, "roc_auc", False )  # leak minimal

    # _run(359931, "mse")
    #

    from sklearn.isotonic import IsotonicRegression

    #     @staticmethod
    #     def _clean_oof_predictions(X, y, feature_metadata, problem_type):
    #         stack_cols = feature_metadata.get_features(required_special_types=['stack'])
    #         if not stack_cols:
    #             return X, y
    #         X = X.copy()
    #         print('yes')
    #         for f in stack_cols:
    #
    #             if problem_type == BINARY:
    #                 reg = IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True)
    #                 X[f] = reg.fit_transform(X[f], y)
    #             elif problem_type == MULTICLASS:
    #                 pass
    #             elif problem_type == REGRESSION:
    #                 pass
    #             else:
    #                 raise ValueError(f"Unsupported Problem Type for cleaning oof predictions! Got: {problem_type}")
    #
    #         return X, y
