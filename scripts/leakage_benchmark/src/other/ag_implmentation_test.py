from shutil import rmtree

import numpy as np
import openml
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from autogluon.tabular import TabularDataset, TabularPredictor


def get_data(tid: int, fold: int):
    # Get Task and dataset from OpenML and return split data
    oml_task = openml.tasks.get_task(tid, download_splits=True, download_data=True,
                                     download_qualities=False, download_features_meta_data=False)

    train_ind, test_ind = oml_task.get_train_test_split_indices(fold)
    X, *_ = oml_task.get_dataset().get_data(dataset_format='dataframe')

    return X.iloc[train_ind, :].reset_index(drop=True), X.iloc[test_ind, :].reset_index(drop=True),\
        oml_task.target_name, oml_task.task_type != 'Supervised Classification'


def _run(task_id, metric):
    print("##### Run for ", task_id)

    l2_train_data, l2_test_data, label, regression = get_data(task_id, 0)

    l2_train_data = l2_train_data.sample(n=min(len(l2_train_data), 10000), random_state=0).reset_index(drop=True)
    l2_test_data = l2_test_data.sample(n=min(len(l2_test_data), 20000), random_state=0).reset_index(drop=True)
    print(l2_train_data.shape, l2_test_data.shape)

    # Run AutoGluon
    print("Start running AutoGluon on L2 data.")
    predictor = TabularPredictor(eval_metric=metric, label=label, verbosity=0, problem_type='binary',
                                 learner_kwargs=dict(random_state=0))
    predictor.fit(
        train_data=l2_train_data,
        hyperparameters={
            "RF": [{}],
            "GBM": [{}],
            # 'XT': [{'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}],
        },
        num_stack_levels=1,
        num_bag_sets=1,
        num_bag_folds=8,
        # presets='best_quality',
        # ag_args_fit=dict(
        #     stack_info_leak_protection=True,
        #                  full_last_weighted_ensemble=False,
        #                  stack_info_leak_protection_protection_normalization=False),
        fit_weighted_ensemble=True,
        ag_args_ensemble={
            # "fold_fitting_strategy": "sequential_local",
            "also_stratify_on_previous_layer": False
        }
    )

    print("##### Results for ", task_id)
    leaderboard = predictor.leaderboard(l2_test_data, silent=True)[['model', 'score_test', 'score_val']].sort_values(by='model').reset_index(drop=True)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard.sort_values(by='score_val', ascending=False))

    # leaderboard = leaderboard[leaderboard['model'] != 'WeightedEnsemble_L3']

    # Treat non-leaking model as L1 model !
    leaderboard['model'] = leaderboard['model'].str.replace('WeightedEnsemble_L2', 'WeightedEnsemble_L1')

    # Get best models per layer
    best_l2_model = leaderboard[~leaderboard['model'].str.endswith('L1')].sort_values(by='score_val', ascending=False).iloc[0].loc['model']
    best_l1_model = leaderboard[leaderboard['model'].str.endswith('L1')].sort_values(by='score_val', ascending=False).iloc[0].loc['model']
    if best_l1_model == 'WeightedEnsemble_L1':
        best_l1_model = 'WeightedEnsemble_L2'
        leaderboard['model'] = leaderboard['model'].str.replace('WeightedEnsemble_L1', 'WeightedEnsemble_L2')

    l1_repo_oof = predictor.predict_proba(l2_train_data, model=best_l1_model, as_multiclass=False)
    l1_oof = predictor.get_oof_pred_proba(model=best_l1_model, as_multiclass=False)

    # predictor.calibrate_decision_threshold(metric='accuracy', model=best_l1_model)
    l1_repo_pred = predictor.predict(l2_train_data, model=best_l1_model)

    l2_repo_oof = predictor.predict_proba(l2_train_data, model=best_l2_model, as_multiclass=False)
    l2_oof = predictor.get_oof_pred_proba(model=best_l2_model, as_multiclass=False)

    # predictor.calibrate_decision_threshold(metric='accuracy', model=best_l2_model)
    l2_repo_pred = predictor.predict(l2_train_data, model=best_l2_model)

    # Get scores
    from sklearn.preprocessing import LabelEncoder
    y = l2_train_data[label]
    y = LabelEncoder().fit_transform(y)
    eval_metric = predictor.eval_metric
    score_l1_repo = eval_metric(y, l1_repo_oof)
    score_l2_repo = eval_metric(y, l2_repo_oof)
    print(f"L1 Repo: {score_l1_repo} | L2 Repo: {score_l2_repo}")
    score_l1_oof = eval_metric(y, l1_oof)
    score_l2_oof = eval_metric(y, l2_oof)
    print(f"L1 OOF: {score_l1_oof} | L2 OOF: {score_l2_oof}")
    score_l1_test = leaderboard.loc[leaderboard['model'] == best_l1_model, 'score_test'].iloc[0]
    score_l2_test = leaderboard.loc[leaderboard['model'] == best_l2_model, 'score_test'].iloc[0]
    print(f"L1 Test: {score_l1_test} | L2 Test: {score_l2_test}")

    # l1 worse val score than l2+
    leak = score_l1_oof < score_l2_oof
    # l2+ worse test score than L1
    leak = leak and (score_l1_test > score_l2_test)

    # -- Potential assumptions that hold for a valid and good stacking models
    #  1) L1 OOF Score  <= L1 Repo Score         # (since L1 repo score leaks and has seen it before)
    #  2) L2 OOF Score  <= L2 Repo Score
    #  3) L1 OOF Score  < L2 OOF Score           # (since the L2 model should perform better)
    #  4) L1 Repo Score < L2 Repo Score          # (since the L2 model seen the data even better or so)

    # -- Current check for leak
    #
    #   Test if: 3) is True and 4 is False
    #       - Idea: 3 holds, as it needs to be the case to leak but 4 does not hold, showing that the
    #           overfitting to the individual fold model predictions has happened.
    #       - Idea: we focused so far to look at a non-leaking models as L1 for reference. Maybe this was a mistake.
    # -----
    #  Not always true because sometimes the repo effect of the leak is not strong enough to affect
    #  test data but instead only better aligns validation and test scores.
    #
    #   Maybe since GES is not cross-validated it can be misleading when it has a weight vector of only
    #    one model. Check for the weight vectors when I am failing. And cross-validated instead.
    #
    #   Maybe add np.isclose to val score comparisons
    #
    #   Maybe sometimes the impact of different splits is too strong even with many repeats
    #

    # l1 worse val score than l2+
    repo_leak = score_l1_oof < score_l2_oof
    # but l2+ has a worse repo score
    repo_leak = repo_leak and ((score_l2_repo < score_l1_repo) or np.isclose(score_l2_repo, score_l1_repo))

    # and predictions must be different?
    repo_leak = repo_leak
    print('All predictions are equal:', all(l1_repo_pred == l2_repo_pred))
    print(f'Leak: {leak} | Repo Leak: {repo_leak}')
    print(f'Idea Correct: {leak == repo_leak}')
    rmtree(predictor.path)
    return leak, repo_leak

if __name__ == '__main__':
    # _run(361339, "roc_auc") # titanic (binary); leak visible
    # _run(189354, "roc_auc") # airlines (binary); leak visible w/o protection (3 folds, 1 repeats)
    # _run(359990, "roc_auc") # albert
    # _run(359983, "roc_auc") # adult

    c_list = [] # 3608, 3913, 359979, 361332, 3918, 3919, 3962
    known_leak = [3899, 43, 3483, 3608, 3616, 3623, 3668, 3684, 3702, 3793, 3904, 3907, 3913, 9971, 167120, 168868, 189354, 190411, 359955, 359956, 359958, 359967, 359972, 359975, 359979, 359983, 359992, 361332, 361334, 361335, 361339,]
    for test_id in known_leak:
        #[
        # 359975,  # leak computation broken, wrong test score used for compare.

        # wrong close
        #3913, 3918]: # , 3964, 361332, 167120, 3968, 3980
         #   3899, 43, 3483, 3608, 3616, 3623, 3668, 3684, 3702, 3793, 3904, 3907, 3913, 9971, 167120, 168868, 189354, 190411, 359955, 359956, 359958, 359967, 359972, 359975, 359979, 359983, 359992, 361332, 361334, 361335, 361339,
         #           3903, 3918, 3919, 3945, 3954, 3962, 3964, 3968, 3976, 3980, 3995, 4000, 9909, 9943, 9959, 9970, 9976, 9983, 14954, 125920, 125968, 146818, 146819, 146820, 168350, 168757, 168911, 189356, 189922, 190137, 190392, 190410, 190412, 359962, 359966, 359968, 359971, 359973, 359980, 359982, 359988, 359989, 359990, 359991, 359994, 360113,]:
        c_list.append(_run(test_id, "roc_auc"))

    c_list = np.array(c_list)
    print('Method ACC:', accuracy_score(c_list[:, 0], c_list[:, 1]))
    cm = pd.DataFrame(confusion_matrix(c_list[:, 0], c_list[:, 1]), columns=['Spotter - False', 'Spotter - True'],
                 index=['GT - False', 'GT - True'])
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(cm)
    # openml_id = 3913 # kc2
    # openml_id = 4000 # OVA_Ovary
    # openml_id = 361331 # GAMETES_Epistasis_2-Way_1000atts_0_4H_EDM-1_EDM-1_1
    # 'GAMETES_Epistasis_2-Way_1000atts_0_4H_EDM-1_EDM-1_1', 'OVA_Ovary',
    # openml_id = 146217 # wine-quality-red (multi-class); leak not visible IMO
    # metric = 'log_loss'

    # openml_id = 359931 # sensory (regression); leak not visible
    # metric = 'mse'



    # Problems:
    # sees leak if leak val exists but not strong enough to hurt tests
    # sometiems duplicates
    # sometimes identical, or very clsoe to identical (or just noise TBH)