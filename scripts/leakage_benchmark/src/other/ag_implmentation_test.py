from shutil import rmtree

import numpy as np
import openml
import pandas as pd
from scipy.special import xlogy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from autogluon.core.metrics import get_metric
from autogluon.tabular import TabularDataset, TabularPredictor


def _log_loss_per_sample(y_true, y_pred):

    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred = y_pred.values if isinstance(y_true, (pd.Series, pd.DataFrame)) else y_pred

    # from sklearn code
    eps = np.finfo(y_pred.dtype).eps
    lb = LabelBinarizer()
    lb.fit(y_true)

    transformed_labels = lb.transform(y_true)

    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(
            1 - transformed_labels, transformed_labels, axis=1
        )
    y_pred = np.clip(y_pred, eps, 1 - eps)

    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    y_pred_sum = y_pred.sum(axis=1)
    y_pred = y_pred / y_pred_sum[:, np.newaxis]
    loss = -xlogy(transformed_labels, y_pred).sum(axis=1)
    return loss
def get_data(tid: int, fold: int):
    # Get Task and dataset from OpenML and return split data
    oml_task = openml.tasks.get_task(tid, download_splits=True, download_data=True,
                                     download_qualities=False, download_features_meta_data=False)

    train_ind, test_ind = oml_task.get_train_test_split_indices(fold)
    X, *_ = oml_task.get_dataset().get_data(dataset_format='dataframe')

    return X.iloc[train_ind, :].reset_index(drop=True), X.iloc[test_ind, :].reset_index(drop=True),\
        oml_task.target_name, oml_task.task_type != 'Supervised Classification'


def _run(task_id, metric):
    l2_train_data, l2_test_data, label, regression = get_data(task_id, 0)
    n_max_cols = 500
    n_max_train_instances = 100000
    n_max_test_instances = 20000

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
        num_bag_sets=4,
        num_bag_folds=8,
        fit_weighted_ensemble=True,
        # presets='best_quality',
        # ag_args_fit=dict(),
        ag_args_ensemble=dict(
            # fold_fitting_strategy="sequential_local"
            nested=True,
            nested_num_folds=8,
        )
    )

    # Get scores
    from sklearn.preprocessing import LabelEncoder
    y = l2_train_data[label]
    y = LabelEncoder().fit_transform(y)
    eval_metric = predictor.eval_metric

    print("### Results for ", task_id)
    leaderboard = predictor.leaderboard(l2_test_data, silent=True)[['model', 'score_test', 'score_val']].sort_values(by='model').reset_index(drop=True)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard.sort_values(by='score_val', ascending=False))

    # leaderboard = leaderboard[leaderboard['model'] != 'WeightedEnsemble_L3']

    # Treat non-leaking model as L1 model !
    non_leaking = ['WeightedEnsemble_L2', 'WeightedEnsemble_BAG_L2']
    for non_leaker in non_leaking:
        leaderboard['model'] = leaderboard['model'].str.replace(non_leaker, non_leaker.replace("L2", "L1"))

    # Get best models per layer
    best_l2_model = leaderboard[~leaderboard['model'].str.endswith('L1')].sort_values(by='score_val', ascending=False).iloc[0].loc['model']
    best_l1_model = leaderboard[leaderboard['model'].str.endswith('L1')].sort_values(by='score_val', ascending=False).iloc[0].loc['model']
    if best_l1_model in [x.replace("L2", "L1") for x in non_leaking]:
        best_l1_model = best_l1_model.replace("L1", "L2")
    for non_leaker in non_leaking:
        leaderboard['model'] = leaderboard['model'].str.replace(non_leaker.replace("L2", "L1"), non_leaker)

    # --- OOF Predictions
    l1_oof = predictor.get_oof_pred_proba(model=best_l1_model, as_multiclass=False)
    l2_oof = predictor.get_oof_pred_proba(model=best_l2_model, as_multiclass=False)

    # --- Reproduction predictions
    # predictor.calibrate_decision_threshold(metric='accuracy', model=best_l1_model)
    l1_repo_pred = predictor.predict(l2_train_data, model=best_l1_model)
    # predictor.calibrate_decision_threshold(metric='accuracy', model=best_l2_model)
    l2_repo_pred = predictor.predict(l2_train_data, model=best_l2_model)

    l1_repo_oof = predictor.predict_proba(l2_train_data, model=best_l1_model, as_multiclass=False,
                                          as_reproduction_predictions_args=dict(y=y))
    l2_repo_oof = predictor.predict_proba(l2_train_data, model=best_l2_model, as_multiclass=False,
                                          as_reproduction_predictions_args=dict(y=y))
    l1_repo_oof_old = predictor.predict_proba(l2_train_data, model=best_l1_model, as_multiclass=False)
    l2_repo_oof_old = predictor.predict_proba(l2_train_data, model=best_l2_model, as_multiclass=False)

    # -- OOF based Reproduction Predictions
    l1_models = [model_name for model_name in leaderboard["model"] if model_name.endswith("BAG_L1")]
    model_name_to_oof = {model_name: predictor.get_oof_pred_proba(model=model_name, as_multiclass=False) for model_name in l1_models}
    l2_true_repo_oof = predictor.predict_proba(l2_train_data, model=best_l2_model, as_multiclass=False,
                                            as_reproduction_predictions_args=dict(y=y, model_name_to_oof=model_name_to_oof))
    l2_true_repo_oof_old = predictor.predict_proba(l2_train_data, model=best_l2_model, as_multiclass=False,
                                            as_reproduction_predictions_args=dict(model_name_to_oof=model_name_to_oof))

    print('### Exp Score overview')
    # workaround to avoid some switch cases for inverse_models only working for L2 internally
    l1_models = [model_name for model_name in leaderboard["model"] if model_name.endswith("BAG_L1")]
    model_name_to_repo_oof = {model_name: predictor.predict_proba(l2_train_data, model=model_name, as_multiclass=False,
                                          as_reproduction_predictions_args=dict(y=y))for model_name in l1_models}
    l2_unseen_repo = predictor.predict_proba(l2_train_data, model=best_l2_model, as_multiclass=False,
                                            as_reproduction_predictions_args=dict(y=y, model_name_to_oof=model_name_to_repo_oof, inverse_models=True))
    score_l2_unseen_repo = eval_metric(y, l2_unseen_repo)
    print(f"L2 Unseen Repo:  {score_l2_unseen_repo:.5f}")

    x = pd.concat([pd.Series(y), l1_oof, l2_oof, l2_unseen_repo, l2_repo_oof], axis=1)
    x.columns = ['label', 'l1_oof', 'l2_oof', 'l2_unseen_repo', 'l2_repo']
    x['diff'] = abs(x['l1_oof'] - x['l2_oof'])
    x['l1_repo'] = l1_repo_oof


    # Monotonic plot

    # adjustment direction
    l1_oof_error = abs(l1_oof - y)
    l2_oof_error = abs(l2_oof - y)
    # l2_unseen_repo_error = abs(l2_unseen_repo - y)

    # False, l2 got worse: True, l2 got better
    # Stacked Overfitting: requires True
    # Stacked Underfitting: requires False
    adjusted_towards_label = l1_oof_error > l2_oof_error

    # Check whether the same model would have made an adjustment in the same direction under
    # optimal (leaking) settings.
    # justified_change = (l2_unseen_repo_error <= l1_oof_error) & (l2_unseen_repo_error <= l1_oof_error)
    # eval_metric_alt = get_metric('log_loss', problem_type='binary')

    # Unjustified change
    oof_false_up = (l1_oof < l2_oof) & (l1_repo_oof > l2_unseen_repo )
    oof_false_down = (l1_oof > l2_oof) & (l1_repo_oof < l2_unseen_repo)
    noise = np.isclose(
        np.min(np.array([l1_oof, l2_oof]), axis=0), np.max(np.array([l1_oof, l2_oof]), axis=0),
        atol=1e-08,  # default
        rtol=0.001  # 0.1% of max abs value
    )
    unjust_change = (oof_false_down | oof_false_up)
    # just_change = (~unjust_change) & (~noise)
    unjust_change = unjust_change & (~noise)

    # maybe factor in that a too large just change is also unjust (e.g. 2x of leakage aware change)


    x['unjust_change'] = unjust_change
    uj_c_ratio = np.mean(unjust_change)
    uj_p_ratio = sum(adjusted_towards_label[unjust_change]/len(unjust_change))
    print(f"Unjust Change %: {uj_c_ratio:.4f}")
    print(f"Positive Unjust Change: {uj_p_ratio:.4f} | "
          f"Negative Unjust Change: {1 - uj_p_ratio:.4f}")
    # print(f"Just Change Error: {eval_metric_alt(y[just_change], l2_oof[just_change])} | "
    #       f"Unjust Change Error: {eval_metric_alt(y[unjust_change], l2_oof[unjust_change])}")

    loss_per_sample_l1 = _log_loss_per_sample(y, l1_oof)
    loss_per_sample_l2 = _log_loss_per_sample(y, l2_oof)

    unjust_positive = unjust_change & adjusted_towards_label
    unjust_negative = unjust_change & (~adjusted_towards_label)

    print(f"Unjust Positive | Error Contribution Change L1 -> L2: {sum(loss_per_sample_l1[unjust_positive])/sum(loss_per_sample_l1):.2f} -> "
          f"{sum(loss_per_sample_l2[unjust_positive])/sum(loss_per_sample_l2):.2f}")
    print(f"Unjust Negative | Error Contribution Change L1 -> L2: {sum(loss_per_sample_l1[unjust_negative])/sum(loss_per_sample_l1):.2f} -> "
          f"{sum(loss_per_sample_l2[unjust_negative])/sum(loss_per_sample_l2):.2f}")

    from scipy.stats import spearmanr
    l1_l2_rs = spearmanr(l1_oof, l2_oof).statistic
    l1_l2_rs_repo = spearmanr(l1_repo_oof, l2_unseen_repo).statistic
    print(f"Normal L1 -> L2 RS {l1_l2_rs:.3f}")
    print(f"Repo   L1 -> L2 RS {l1_l2_rs_repo:.3f}")

    tmp_res = dict(
        l1_l2_rs=l1_l2_rs, l1_l2_rs_repo=l1_l2_rs_repo,  uj_c_ratio=uj_c_ratio, uj_p_ratio = uj_p_ratio,
        uj_n_ratio=1-uj_p_ratio
    )

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.scatterplot(x=l1_oof.values, y=l2_oof.values)
    # sns.scatterplot(x=l1_repo_oof.values, y=l2_unseen_repo.values)
    # plt.show()

    # x['adjusted_towards_label'] = adjusted_towards_label
    # x['justified_change'] = justified_change
    # x['leaking'] = adjusted_towards_label & (~justified_change)
    # exit()
    # # Get subset of leak-free instances:
    # print('### Subset Score overview')
    # no_leak_instances = ~x['leaking']
    # # leak_instances_per = np.mean(~no_leak_instances)
    #
    # score_no_leak_l1_repo = eval_metric_alt(y[no_leak_instances], l1_repo_oof[no_leak_instances])
    # score_no_leak_l2_repo = eval_metric_alt(y[no_leak_instances], l2_repo_oof[no_leak_instances])
    # score_no_leak_l2_true_repo = eval_metric_alt(y[no_leak_instances], l2_true_repo_oof[no_leak_instances])
    # print(f"L1 NL-Repo:  {score_no_leak_l1_repo:.5f} | L2 NL-Repo: \t {score_no_leak_l2_repo:.5f} | L2 NL-T-Repo: \t {score_no_leak_l2_true_repo:.5f}")
    #
    # score_no_leak_l1_oof = eval_metric_alt(y[no_leak_instances], l1_oof[no_leak_instances])
    # score_no_leak_l2_oof = eval_metric_alt(y[no_leak_instances], l2_oof[no_leak_instances])
    # print(f"L1 NL-OOF: \t {score_no_leak_l1_oof:.5f} | L2 NL-OOF: \t {score_no_leak_l2_oof:.5f}")

    print('### Score overview')
    score_l1_repo = eval_metric(y, l1_repo_oof)
    score_l2_repo = eval_metric(y, l2_repo_oof)
    score_l2_true_repo = eval_metric(y, l2_true_repo_oof)
    print(f"L1 Repo: \t {score_l1_repo:.5f} | L2 Repo: \t {score_l2_repo:.5f} | L2 T-Repo:   \t {score_l2_true_repo:.5f}")
    score_l1_repo_old = eval_metric(y, l1_repo_oof_old)
    score_l2_repo_old = eval_metric(y, l2_repo_oof_old)
    score_l2_true_repo_old = eval_metric(y, l2_true_repo_oof_old)
    print(f"L1 O-Repo: \t {score_l1_repo_old:.5f} | L2 O-Repo: \t {score_l2_repo_old:.5f} | L2 O-T-Repo: \t {score_l2_true_repo_old:.5f}")
    score_l1_oof = eval_metric(y, l1_oof)
    score_l2_oof = eval_metric(y, l2_oof)
    print(f"L1 OOF: \t {score_l1_oof:.5f} | L2 OOF: \t\t {score_l2_oof:.5f}")
    score_l1_test = leaderboard.loc[leaderboard['model'] == best_l1_model, 'score_test'].iloc[0]
    score_l2_test = leaderboard.loc[leaderboard['model'] == best_l2_model, 'score_test'].iloc[0]
    print(f"L1 Test: \t {score_l1_test:.5f} | L2 Test: \t {score_l2_test:.5f}")

    # l1 worse val score than l2+
    leak = score_l1_oof < score_l2_oof
    # l2+ worse test score than L1
    leak = leak and (score_l1_test >= score_l2_test)

    # -- Potential assumptions that hold for a valid and good stacking models
    #  1) L1 OOF Score  <= L1 Repo Score         # (since L1 repo score leaks and has seen it before)
    #  2) L2 OOF Score  <= L2 Repo Score
    #  3) L1 OOF Score  < L2 OOF Score           # (since the L2 model should perform better)
    #  4) L1 Repo Score <= L2 Repo Score          # (since the L2 model seen the data even better or so)
    #       - This does not really hold due to how stacking works IMO
    #
    # -- Introduce holdout free repo score
    # Idea: only predict with models that have seen the data before. Producing an HF-Repo Score.
    #  5) L1 HF-Repo Score <= L2 HF-Repo score
    #  6) L1 Repo Score <= HF-Repo Score         # must be!
    #  7) L2 HF-Repo Score >= L2 Repo Score
    #       - If False, then the leak must have happened! (but we have a counter example, how?)

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
    #   Idea problem: this idea tells us that the validation score is not representative when comparing L1 and L2
    #       but this does not mean that the test score is worse. E.g. we could jump from 0.9 to 0.95 val score but
    #       stay at 0.912 test score. This would trigger the method. Shows a change in the gap between val and test
    #       going from L1 to L2.

    # l1 worse val score than l2+
    repo_leak = score_l1_oof < score_l2_oof
    # repo_leak = repo_leak and (score_no_leak_l1_oof > score_no_leak_l2_oof)

    repo_leak = repo_leak and (
        # but l2+ has a worse repo score (or a very similar score)
        (score_l2_repo < score_l1_repo)
        # or np.isclose(score_l1_repo, score_l2_repo)
    )

    # Avoid optimum making isclose misguided
    # repo_leak = repo_leak and ((score_l1_repo != eval_metric.optimum) or (score_l2_repo != eval_metric.optimum))
    # repo_leak = repo_leak and not (np.isclose(score_l1_repo, eval_metric.optimum) and np.isclose(score_l2_repo, eval_metric.optimum))

    # closeness protection (i.e., leak requires reasonable increase in val score to have an impact)
    # repo_leak = repo_leak and  (not np.isclose(score_l2_oof, score_l1_oof, atol=1e-5, rtol=1e-2))
    # print('Is close?:', np.isclose(score_l2_oof, score_l1_oof, atol=1e-5, rtol=1e-2))

    stacking_is_better = (score_l1_oof < score_l2_oof) and (score_l1_test < score_l2_test)
    eval_metric = get_metric('log_loss', problem_type='binary')
    print(f"## Alt Metrics")
    am_score_l1_repo = eval_metric(y, l1_repo_oof)
    am_score_l2_repo = eval_metric(y, l2_repo_oof)
    am_score_l2_true_repo = eval_metric(y, l2_true_repo_oof)
    print(f"[ALT METRIC] L1 Repo: \t {am_score_l1_repo:.5f} | L2 Repo: \t {am_score_l2_repo:.5f} | L2 T-Repo:   \t {am_score_l2_true_repo:.5f}")
    am_score_l1_oof = eval_metric(y, l1_oof)
    am_score_l2_oof = eval_metric(y, l2_oof)
    print(f"[ALT METRIC] L1 OOF: \t {am_score_l1_oof:.5f} | L2 OOF: \t {am_score_l2_oof:.5f}")

    print(f"## Overview")
    print(f'Stacking is better: {stacking_is_better}')
    print(f'Leak: {leak} | Repo Leak: {repo_leak}')
    print(f'Idea Correct: {leak == repo_leak}')
    rmtree(predictor.path)
    return dict(
        task_id=task_id,
        l1_repo=score_l1_repo, l2_repo=score_l2_repo, l2_true_repo=score_l2_true_repo,
        l1_repo_old=score_l1_repo_old, l2_repo_old=score_l2_repo_old, l2_true_repo_old=score_l2_true_repo_old,
        l1_val=score_l1_oof, l2_val=score_l2_oof, l1_test=score_l1_test, l2_test=score_l2_test,
        # l1_repo_no_leak=score_no_leak_l1_repo, l2_repo_no_leak=score_no_leak_l2_repo,
        # l1_oof_no_leak=score_no_leak_l1_oof, l2_oof_no_leak=score_no_leak_l2_oof,
        #  l2_true_repo_no_leak=score_no_leak_l2_true_repo,
        # leak_instances_per=leak_instances_per,
        leak=leak, leak_spotted=repo_leak,
        stacking_is_better=stacking_is_better, stacking_has_no_impact=score_l1_oof >= score_l2_oof,
        all_predictions_equal=all(l1_repo_pred == l2_repo_pred),
        am_l1_repo = am_score_l1_repo, am_l2_repo=am_score_l2_repo, am_l2_true_repo=am_score_l2_true_repo,
        am_l1_val=am_score_l1_oof, am_l2_val=am_score_l2_oof,
        **tmp_res
    )


if __name__ == '__main__':
    c_list = []
    all_tids = [43, 3483, 3608, 3616, 3623, 3668, 3684, 3702, 3793, 3904, 3907, 3913, 9971, 167120, 168868, 189354,
                190411, 359955, 359956, 359958, 359967, 359972, 359975, 359979, 359983, 359992, 361332, 361334,
                361335, 361339, 37, 219, 3581, 3583, 3591, 3593, 3600, 3601, 3606, 3618, 3627, 3664, 3667, 3672, 3681,
                3698, 3704, 3712, 3735, 3747, 3749, 3764, 3766, 3783, 3786, 3799, 3800, 3812, 3844, 3892, 3899,
                3903, 3918, 3919, 3945, 3954, 3962,  3968, 3976, 3980, 3995, 4000, 9909, 9943, 9959, 9970,
                9976, 9983, 14954, 125920, 125968, 146818, 146819, 146820, 168350, 168757, 168911, 189356, 3688, 3690,
                189922, 190137, 190392, 190410, 190412, 359962, 359966, 359968, 359971, 359973, 359980, 359982,
                359988, 359989, 359990, 359991, 359994, 360113, 360114, 360975, 361331, 361333, 361336, 361340,
                361341, 361342]

    to_test = [ 3904,  # leak

        219,  9909,  37, # stacking is better

        3904, 3964, 3483, 3623 , 3913, 9971, 167120, 168868, 189354, 359955, 359975, 359979, 359992, 361339,3593,
               3735, 3812, 3892, 3903, 3918, 3945, 3962, 4000, 125920, 168911, 359962, 359966, 359971, 359991,
               359994, 360113, 360975, 361333, 361336, 361341
                ]
    to_test = all_tids
    import pickle

    for en_idx, test_id in enumerate(to_test, start=1):
        print(f"\n##### Run for {test_id} ({en_idx}/{len(to_test)})")
        c_list.append(_run(test_id, "roc_auc"))

        with open(f'results_nested.pkl', 'wb') as f:
            pickle.dump(c_list, f)

    # --- Other
    # _run(361339, "roc_auc") # titanic (binary); leak visible
    # _run(189354, "roc_auc") # airlines (binary); leak visible w/o protection (3 folds, 1 repeats)
    # _run(359990, "roc_auc") # albert
    # _run(359983, "roc_auc") # adult
    # openml_id = 3913 # kc2
    # openml_id = 4000 # OVA_Ovary
    # openml_id = 361331 # GAMETES_Epistasis_2-Way_1000atts_0_4H_EDM-1_EDM-1_1
    # 'GAMETES_Epistasis_2-Way_1000atts_0_4H_EDM-1_EDM-1_1', 'OVA_Ovary',
    # openml_id = 146217 # wine-quality-red (multi-class); leak not visible IMO
    # metric = 'log_loss'
    # openml_id = 359931 # sensory (regression); leak not visible
    # metric = 'mse'
