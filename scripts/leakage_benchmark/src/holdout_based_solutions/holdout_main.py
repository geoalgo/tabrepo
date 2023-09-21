from functools import partial

import numpy as np

from scripts.leakage_benchmark.src.holdout_based_solutions.ag_test_utils import (
    get_data, inspect_full_results, inspect_leaderboard,
    print_and_get_leaderboard, sub_sample)
from scripts.leakage_benchmark.src.holdout_based_solutions.holdout_approaches import (
    default, use_holdout)
from scripts.leakage_benchmark.src.holdout_based_solutions.stacked_overfitting_proxy_model import \
    stacked_overfitting_proxy_model

BASE_SEED = 239785


def _run(task_id, metric):
    train_data, test_data, label, regression = get_data(task_id, 0)
    train_data, test_data, label = sub_sample(train_data, test_data, label, n_max_cols=50, n_max_train_instances=10000, n_max_test_instances=2000)

    # --- Determine whether to use stacking by proxy ---
    rng = np.random.RandomState(BASE_SEED)
    holdout_seed = rng.randint(0, 2**32)
    use_stacking_opinions = []
    for _ in range(5):
        use_stacking_opinions.append(stacked_overfitting_proxy_model(train_data, label, split_random_state=rng.randint(0, 2**32)))
    print(f"Proxy Opinions: {use_stacking_opinions}")
    proxy_found_so = any(use_stacking_opinions)

    # --- AutoGluon Specification ---
    predictor_para = dict(eval_metric=metric, label=label, verbosity=0, problem_type="binary", learner_kwargs=dict(random_state=0))
    fit_para = dict(
        hyperparameters={
            "RF": [{}],
            "GBM": [{}],
            # 'XT': [{'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}],
        },
        num_stack_levels=1,
        num_bag_sets=1,
        num_bag_folds=8,
        fit_weighted_ensemble=True,
        # presets='best_quality',
        # ag_args_fit=dict(),
        ag_args_ensemble=dict(
            # fold_fitting_strategy="sequential_local"
            # nested=True,  # wrong branch for this currently!
            # nested_num_folds=8,
        ),
    )

    res = dict(task_id=task_id, results={}, leaderboards={})
    for method_func in [
        # Default AutoGluon with Stacking
        partial(default, use_stacking=True),
        # Default AutoGluon without stacking
        partial(default, use_stacking=False),
        # Select final model based on holdout
        partial(use_holdout, refit_autogluon=False, select_on_holdout=True),
        # Select based on holdout and refit
        partial(use_holdout, refit_autogluon=True, select_on_holdout=True),
        # Determine SO based on holdout and refit with or without stacking
        partial(use_holdout, refit_autogluon=True, dynamic_stacking=True),
        # Determine GES Weights based on holdout, then refit and use these weights for the final predictions.
        partial(use_holdout, refit_autogluon=True, ges_holdout=True),
    ]:
        print("\n")
        predictor, method_name, corrected_val_scores = method_func(train_data, label, fit_para, predictor_para, holdout_seed=holdout_seed)
        leaderboard = print_and_get_leaderboard(predictor, test_data, method_name, corrected_val_scores)
        res["leaderboards"][method_name] = leaderboard
        res["results"][method_name] = inspect_leaderboard(leaderboard, predictor.get_model_best())

    inspect_full_results(res, proxy_found_so)

    return res


if __name__ == "__main__":
    c_list = []
    all_tids = [
        3483,
        43,
        3608,
        3616,
        3623,
        3668,
        3684,
        3702,
        3793,
        3904,
        3907,
        3913,
        9971,
        167120,
        168868,
        189354,
        190411,
        359955,
        359956,
        359958,
        359967,
        359972,
        359975,
        359979,
        359983,
        359992,
        361332,
        361334,
        361335,
        361339,
        37,
        219,
        3581,
        3583,
        3591,
        3593,
        3600,
        3601,
        3606,
        3618,
        3627,
        3664,
        3667,
        3672,
        3681,
        3698,
        3704,
        3712,
        3735,
        3747,
        3749,
        3764,
        3766,
        3783,
        3786,
        3799,
        3800,
        3812,
        3844,
        3892,
        3899,
        3903,
        3918,
        3919,
        3945,
        3954,
        3962,
        3968,
        3976,
        3980,
        3995,
        4000,
        9909,
        9943,
        9959,
        9970,
        9976,
        9983,
        14954,
        125920,
        125968,
        146818,
        146819,
        146820,
        168350,
        168757,
        168911,
        189356,
        3688,
        3690,
        189922,
        190137,
        190392,
        190410,
        190412,
        359962,
        359966,
        359968,
        359971,
        359973,
        359980,
        359982,
        359988,
        359989,
        359990,
        359991,
        359994,
        360113,
        360114,
        360975,
        361331,
        361333,
        361336,
        361340,
        361341,
        361342,
    ]

    import pickle

    for en_idx, test_id in enumerate(all_tids, start=1):
        print(f"##### Run for {test_id} ({en_idx}/{len(all_tids)})")
        c_list.append(_run(test_id, "roc_auc"))

        with open(f"results_curr.pkl", "wb") as f:
            pickle.dump(c_list, f)
        print("\n\n")

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
