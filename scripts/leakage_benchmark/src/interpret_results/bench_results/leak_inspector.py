import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def fold_avg_leak(lb, i_f_name, f_name):
    # ------- Leak analysis (wrong across folds)
    # select for stack
    lb_info_leak = lb[lb["framework_parent"].isin([i_f_name])]
    # ignore weighted ensembles
    lb_info_leak = lb_info_leak[~lb_info_leak["model"].isin(["autogluon_ensemble", "autogluon_single", "WeightedEnsemble_BAG_L2"])]
    lb_info_leak.loc[lb_info_leak["model"] == "WeightedEnsemble_L3", "stack_level"] = 2
    lb_info_leak.loc[lb_info_leak["model"] == "WeightedEnsemble_L2", "stack_level"] = 1

    # select by best performing model according to val score per layer
    lb_info_leak = lb_info_leak.loc[lb_info_leak.groupby(by=["dataset", "stack_level"])["score_val"].idxmax()]

    # select only those cases where l2 is better than l1 according to val score
    leak_candidates_val_score = lb_info_leak.loc[lb_info_leak.groupby("dataset")["score_val"].idxmax()]
    leak_candidates_val_score = leak_candidates_val_score[leak_candidates_val_score["stack_level"] != 1]

    # select only those were the l1 is better than l2 according to test score
    leak_candidates_test_score = lb_info_leak.loc[lb_info_leak.groupby("dataset")["metric_score"].idxmax()]
    leak_candidates_test_score = leak_candidates_test_score[leak_candidates_test_score["stack_level"] == 1]

    # get the intersection
    leak_datasets = set(leak_candidates_test_score["dataset"]).intersection(set(leak_candidates_val_score["dataset"]))

    EPS = np.finfo(np.float32).eps
    leak_l1_error = leak_candidates_test_score[leak_candidates_test_score["dataset"].isin(leak_datasets)][["dataset", "metric_error"]].set_index("dataset")
    leak_l2_error = leak_candidates_val_score[leak_candidates_val_score["dataset"].isin(leak_datasets)][["dataset", "metric_error"]].set_index("dataset")
    leak_l1_error["metric_error"] += EPS
    leak_l2_error["metric_error"] += EPS
    equal_mask = (leak_l2_error - leak_l1_error) == 0

    test_score_loss_by_leak = (leak_l2_error - leak_l1_error) / leak_l2_error

    print(f_name, leak_datasets, len(leak_datasets))
    print(test_score_loss_by_leak.describe())


def fit_curve(lb):
    lb = lb.copy()
    lb = lb.loc[lb.groupby(["framework_parent", "dataset"])["score_val"].idxmax()]
    lb["log_val_minus_test_diff"] = lb["score_val"] - lb["metric_score"]

    map_dict = (
        lb.loc[lb["framework_parent"] == "AG_no_stack_4h8c", ["log_val_minus_test_diff", "dataset"]]
        .sort_values(by="log_val_minus_test_diff")
        .reset_index(drop=True)["dataset"]
        .to_dict()
    )
    map_dict = {v: k for k, v in map_dict.items()}
    lb["Sorted Index"] = lb["dataset"].map(map_dict)
    plot_df = lb[lb["framework_parent"].isin(["AG_stack_4h8c", "AG_no_stack_4h8c"])]

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.lineplot(data=plot_df, x="Sorted Index", y="log_val_minus_test_diff", hue="framework_parent")
    # plt.yscale('symlog')
    plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0,
        fancybox=True, shadow=True, ncol=3
    )
    plt.tight_layout()
    plt.show()
    print()
