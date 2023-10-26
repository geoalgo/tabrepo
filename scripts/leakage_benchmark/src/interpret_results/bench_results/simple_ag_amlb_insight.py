import glob
import json
import pathlib
import pickle
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from scipy.stats import wilcoxon

from scripts.leakage_benchmark.src.interpret_results.bench_results import \
    leak_inspector
from scripts.leakage_benchmark.src.interpret_results.plotting.cd_plot import \
    cd_evaluation

res = pd.read_csv("./input_data/results_preprocessed.csv")
lb = pd.read_csv("./input_data/leaderboard_preprocessed.csv")
lb_n = pd.read_csv("./input_data/leaderboard_preprocessed2023_09_28_purucker.csv")
res_n = pd.read_csv("./input_data/results_preprocessed2023_09_28_purucker.csv")
problem_types = ["binary"]
lb = pd.concat([lb, lb_n])
res = pd.concat([res, res_n])

res = res
lb = lb[lb["problem_type"].isin(problem_types)]
res = res[res["problem_type"].isin(problem_types)]
filter_methods = ['AG_stacking_proxy_v2_4h8c', 'AG_stacking_proxy_v2_1h8c', 'AG_stack_clean_oof_4h8c', 'AG_stack_clean_oof_1h8c', 'AG_stack_ho_dynamic_clean_oof_4h8c', 'AG_stack_ho_dynamic_clean_oof_1h8c']

for f_m in filter_methods:
    res = res[~res['framework'].str.startswith(f_m)]
    lb = lb[~lb['framework_parent'].str.startswith(f_m)]

fold_missing = res.groupby(by=["dataset", "framework"])['fold'].count() != 10
missing_fold_map = {}
framework_missing_count = {}
for m_dataset, m_framework in list(fold_missing.index[fold_missing].to_frame().to_dict()['dataset'].keys()):
    missing_subset = res[(res.dataset == m_dataset) & (res.framework == m_framework)]
    missing_folds = set(range(10)) - set(missing_subset['fold'])
    missing_fold_map[(m_dataset, m_framework)] = missing_folds
    if m_framework not in framework_missing_count:
        framework_missing_count[m_framework] = 0
    framework_missing_count[m_framework] += len(missing_folds)

print(f"The following folds are missing: {missing_fold_map}")
print(f"Missing per Framework count: {framework_missing_count}")

print(f"Aggregating over all folds ignoring missing folds...")
res = res.groupby(by=["dataset", "framework"]).mean().reset_index().drop(columns=["fold"])
lb = lb.groupby(by=["dataset", "framework", "model", "framework_parent", "stack_level"]).mean().reset_index().drop(columns=["fold"])

# Avg number of models per framework
print("Model Count:", res.groupby("framework")["models_count"].mean())
print("Ensemble Size:", res.groupby("framework")["models_ensemble_count"].mean())

l3_occ_map = {}
for f_name in sorted(set(res["framework"])):
    i_f_name = lb.loc[lb["framework"] == f_name + "_autogluon_ensemble", "framework_parent"].iloc[0]
    l3_occ_map[f_name.split("2023")[0][:-1]] = np.mean(lb[lb["framework_parent"].isin([i_f_name])].groupby("dataset")["stack_level"].max() == 3)

leak_inspector.fit_curve(lb )


def win_rate(in_df, in_model):
    rank_df = in_df.rank(axis=1, ascending=False)
    w = sum(rank_df[in_model] == 1)
    t = sum(rank_df[in_model] == 1.5)
    l = sum(rank_df[in_model] == 2)
    return (w + 0.5 * t) / len(rank_df), w, t, l

def _to_sig_symbol(p):
    # Assume default alpha of 0.05

    if p > 0.05:
        return "ns"

    if p <= 0.0001:
        return "****"

    if p <= 0.001:
        return "***"

    if p <= 0.01:
        return "**"

    if p <= 0.05:
        return "*"

baseline = "AG_no_stack_4h8c"

# Stitch together new models
for metric in ["metric_score"]:
    res['framework'] = res['framework'].apply(lambda x: x.split("2023")[0][:-1])
    # res = res[~res['framework'].isin(['AG_stacking_proxy_v2_4h8c', 'AG_stacking_proxy_v2_1h8c', 'AG_stack_clean_oof_4h8c', 'AG_stack_clean_oof_1h8c',
    #                                   'AG_stack_ho_dynamic_clean_oof_4h8c', 'AG_stack_ho_dynamic_clean_oof_1h8c',
    #
    #                                   # "AG_stack_clean_oof_v2_1h8c", 'AG_stack_clean_oof_v2_4h8c', 'AG_stack_ho_dynamic_clean_oof_v2_4h8c','AG_stack_ho_dynamic_clean_oof_v2_1h8c'
    #                                   ])]

    performance_per_dataset = res.pivot(index="dataset", columns="framework", values=metric)
    for f_work in performance_per_dataset.columns:
        print(f"Failures for {f_work}: {list(performance_per_dataset.index[performance_per_dataset[f_work].isnull()])}")

    performance_per_dataset = performance_per_dataset.dropna()
    # performance_per_dataset = performance_per_dataset.round(4)
    lb["diff"] = lb["score_val"] - lb["metric_score"]
    lb[(lb['dataset'] == 'blood-transfusion-service-center') & (lb['framework_parent'].isin(["AG_no_stack_4h8c", "AG_stack_clean_oof_v3_4h8c"]))][
        ["framework", "model", "stack_level", "metric_score", "score_val", 'diff']]
    # performance_per_dataset[performance_per_dataset.idxmax(axis=1).str.startswith('AG_no_stack')][["AG_ho_dynamic_stacking_4h8c", "AG_no_stack_4h8c", "AG_stack_clean_oof_v2_4h8c", "AG_stack_clean_oof_v3_4h8c"]]
    # lb[(lb['dataset'] == 'blood-transfusion-service-center') & (lb['framework_parent'].isin(["AG_no_stack_4h8c", "AG_stack_clean_oof_v2_4h8c"]))][["framework", "model", "stack_level", "metric_score", "score_val"]]
    tmp_res = []
    train_times = res.pivot(index="dataset", columns="framework", values="time_train_s").dropna().sum()



    for model in list(performance_per_dataset):
        l3_occ = l3_occ_map[model]
        train_time_rel = train_times[model] / train_times[baseline]
        if model == baseline:
            tmp_res.append(("[REFERENCE] " + model, 0.5, '-', 0, len(performance_per_dataset), 0, l3_occ, train_time_rel))
        else:
            tmp_r, tmp_w, tmp_t, tmp_l = win_rate(performance_per_dataset[[model, baseline]], model)
            _, tmp_p = wilcoxon(performance_per_dataset[model], performance_per_dataset[baseline], method='approx')
            tmp_res.append([model, tmp_r, _to_sig_symbol(tmp_p), tmp_w, tmp_t, tmp_l, l3_occ, train_time_rel])

    tmp_res = pd.DataFrame(tmp_res, columns=["model", "ratio", "significance", "win", "tie", "lose", "% L3 Trained", "Rel Train Time (x)"]).sort_values(
        by="ratio", ascending=False
    )
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print("\n\n", tmp_res, "\n\n")

    # cd_evaluation(performance_per_dataset, True, None, ignore_non_significance=True, plt_title=f"{metric} | {len(performance_per_dataset)}")
