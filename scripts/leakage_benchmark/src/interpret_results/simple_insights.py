import pathlib
import pickle

from scripts.leakage_benchmark.src.config_and_data_utils import LeakageBenchmarkFoldResults, LeakageBenchmarkResults
from scripts.leakage_benchmark.src.interpret_results.plotting.cd_plot import cd_evaluation
from scripts.leakage_benchmark.src.interpret_results.plotting.distribution_plot import _distribution_plot, normalized_improvement_distribution_plot
from typing import List
import pandas as pd
import glob

BASE_PATH = pathlib.Path(__file__).parent.parent.parent.resolve() / 'output'


def _read_fold_results() -> [List[LeakageBenchmarkResults], List[List[LeakageBenchmarkFoldResults]]]:
    file_dir = BASE_PATH / 'fold_results_per_dataset'
    data: List[List[LeakageBenchmarkFoldResults]] = []
    res: List[LeakageBenchmarkResults] = []

    for f_path in glob.glob(str(file_dir / 'fold_results_*.pkl')):
        with open(f_path, 'rb') as f:
            fold_data = pickle.load(f)
            if not fold_data:
                raise ValueError(f"Empty fold data in {f_path}")
            data.append(fold_data)

    # Aggregate over folds
    for dataset_res in data:
        res.append(LeakageBenchmarkResults.aggregate_fold_results(dataset_res))

    return res, data


def _run():
    fig_dir = BASE_PATH / 'figures'
    fig_dir.mkdir(exist_ok=True, parents=True)

    res, data = _read_fold_results()
    l2_models = res[0].l2_models
    all_res = pd.concat([r.results_df for r in res])
    performance_per_dataset = pd.concat(
        [r.leak_overview_df.drop(index=['all_l2_models'])[['simulate_test_score']].T.rename(
            index=dict(simulate_test_score=r.dataset)) for r in res])
    performance_per_dataset.index.name = 'Dataset'

    for task_type in all_res['problem_type'].unique():
        print(f"### Task type: {task_type}")
        task_res = all_res[all_res['problem_type'] == task_type]
        # Plot leakage prevention quality (overall)
        print('Overall Plot')
        (fig_dir / task_type).mkdir(exist_ok=True, parents=True)
        cd_evaluation(performance_per_dataset[performance_per_dataset.index.isin(task_res['dataset'])], True,
                      fig_dir / task_type / 'leakage_mitigation_all_compare_cd_plot.pdf',
                      ignore_non_significance=True)
        normalized_improvement_distribution_plot(performance_per_dataset[performance_per_dataset.index.isin(task_res['dataset'])], True,
                                                 'LightGBM_noise_dummy_BAG_L2', fig_dir / task_type / 'leakage_mitigation_all_compare_ni_plot.pdf')

        for leak_baseline in l2_models:
            print(f'\n## For Method: {leak_baseline}')
            datasets_that_leak = task_res.loc[task_res[f'relative_test_loss_by_leak/{leak_baseline}'] > 0, 'dataset']
            loss_for_leak_ds = task_res[task_res['dataset'].isin(datasets_that_leak)][
                f'relative_test_loss_by_leak/{leak_baseline}']
            gap_for_leak_ds = task_res[task_res['dataset'].isin(datasets_that_leak)][
                f'misfit_gap_measure/{leak_baseline}']

            print(f"The following {len(list(datasets_that_leak))}/{len(task_res)} datasets leak for {leak_baseline}",
                  list(datasets_that_leak),
                  f"\nFor these datasets, the leak increases the error from min {loss_for_leak_ds.min() * 100:.3f}% "
                  f"to max {loss_for_leak_ds.max() * 100:.3f}% (avg.: {loss_for_leak_ds.mean() * 100:.3f}%).",
                  f"\nMoreover, the gap between validation and test loss increases on average by {gap_for_leak_ds.mean() * 100:.3f}%.")

            # if len(datasets_that_leak) < 3:
            #     print('Not enough dataset that leak for CD plots!')
            #     continue
            # # Plot this leaks only
            # cd_evaluation(performance_per_dataset[performance_per_dataset.index.isin(datasets_that_leak)], True,
            #               fig_dir / task_type / 'leakage_mitigation_leak_compare_cd_plot.pdf',
            #               ignore_non_significance=True)
            #
            # # Plot no leaks only
            # cd_evaluation(performance_per_dataset[~performance_per_dataset.index.isin(datasets_that_leak)], True,
            #               fig_dir / task_type / 'leakage_mitigation_no_leak_compare_cd_plot.pdf',
            #               ignore_non_significance=True)

        # # Stat plots
        # task_res['state'] = 'X'
        # task_res.loc[task_res.dataset.isin(datasets_that_leak), 'state'] = 'leak'
        #
        # no_compare_att = ['folds', 'state', 'dataset', 'eval_metric_name', 'problem_type']
        # no_compare_att += task_res.columns[task_res.isna().any()].tolist()
        #
        # for att in all_res.columns:
        #     if att in no_compare_att:
        #         continue
        #
        #     _distribution_plot(task_res,
        #                        x_col=att, y_col="state",
        #                        x_label=att, y_label="State",
        #                        save_path=None,
        #                        baseline_val=task_res[att].mean(),
        #                        overwrite_xlim=None,
        #                        xlim_max=None,
        #                        baseline_name="Average All Dataset",
        #                        dot_name=f"{att}",
        #                        sort_by=None,
        #                        figsize=(12, 10))

    # # Plot leakage prevention quality (overall)
    # cd_evaluation(performance_per_dataset, True,
    #               fig_dir / 'leakage_mitigation_all_compare_cd_plot.pdf',
    #               ignore_non_significance=True)
    #
    # # Plot leaks only
    # cd_evaluation(performance_per_dataset[performance_per_dataset.index.isin(overall_datasets_that_leak)], True,
    #               fig_dir / 'leakage_mitigation_leak_compare_cd_plot.pdf', ignore_non_significance=True)

    # # --- Overall compare for all shared cols
    # print(f"Overall {len(overall_datasets_that_leak)}/{len(all_res)} Datasets Leak")
    # all_res['state'] = 'X'
    # # all_res.loc[all_res.dataset.isin(overall_datasets_that_leak), 'state'] = 'leak'
    #
    # no_compare_att = ['folds', 'state', 'dataset', 'eval_metric_name', 'problem_type']
    # no_compare_att += all_res.columns[all_res.isna().any()].tolist()
    # for att in all_res.columns:
    #     if att in no_compare_att:
    #         continue
    #
    #     _distribution_plot(all_res,
    #                        x_col=att, y_col="state",
    #                        x_label=att, y_label="State",
    #                        save_path=None,
    #                        baseline_val=all_res[att].mean(),
    #                        overwrite_xlim=None,
    #                        xlim_max=None,
    #                        xlim=False,
    #                        baseline_name="Average All Dataset",
    #                        dot_name=f"{att}",
    #                        sort_by=None,
    #                        figsize=(12, 10))


if __name__ == '__main__':
    _run()
