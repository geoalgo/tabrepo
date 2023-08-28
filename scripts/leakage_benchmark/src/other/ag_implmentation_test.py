import pandas as pd
import openml
from autogluon.tabular import TabularPredictor


openml_id = 189354  # airlines (binary); leak visible w/o protection (3 folds, 1 repeats)
metric = "roc_auc"

# openml_id = 146217 # wine-quality-red (multi-class); leak not visible IMO
# metric = 'log_loss'

# openml_id = 359931 # sensory (regression); leak not visible
# metric = 'mse'

def get_data(tid: int, fold: int):
    # Get Task and dataset from OpenML and return split data
    oml_task = openml.tasks.get_task(tid, download_splits=True, download_data=True,
                                     download_qualities=False, download_features_meta_data=False)

    train_ind, test_ind = oml_task.get_train_test_split_indices(fold)
    X, *_ = oml_task.get_dataset().get_data(dataset_format='dataframe')

    return X.iloc[train_ind, :].reset_index(drop=True), X.iloc[test_ind, :].reset_index(drop=True),\
        oml_task.target_name, oml_task.task_type != 'Supervised Classification'


def _run():
    l2_train_data, l2_test_data, label, regression = get_data(openml_id, 0)

    l2_train_data = l2_train_data.sample(n=min(len(l2_train_data), 10000), random_state=0).reset_index(drop=True)
    l2_test_data = l2_test_data.sample(n=min(len(l2_test_data), 20000), random_state=0).reset_index(drop=True)

    # Run AutoGluon
    print("Start running AutoGluon on L2 data.")
    predictor = TabularPredictor(eval_metric=metric, label=label, verbosity=2,
                                 learner_kwargs=dict(random_state=0))
    predictor.fit(
        train_data=l2_train_data,
        hyperparameters={
            "RF": [{}],
            "GBM": [{}],
        },
        num_stack_levels=1,
        num_bag_folds=8,
        ag_args_fit=dict(stack_info_leak_protection=True),
        # ag_args_ensemble={"fold_fitting_strategy": "sequential_local"}
    )

    leaderboard = predictor.leaderboard(l2_test_data, silent=True)[['model', 'score_test', 'score_val']].sort_values(by='model').reset_index(drop=True)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)


if __name__ == '__main__':
    _run()
