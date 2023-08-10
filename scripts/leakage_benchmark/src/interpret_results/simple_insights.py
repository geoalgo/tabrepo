import pathlib
import pickle

from scripts.leakage_benchmark.src.config_and_data_utils import LeakageBenchmarkFoldResults, LeakageBenchmarkResults
from scripts.leakage_benchmark.src.interpret_results.plotting.cd_plot import cd_evaluation
from typing import List

import pandas as pd


def _run():
    file_dir = pathlib.Path(__file__).parent.parent.parent.resolve() / 'output'
    fig_dir = file_dir / 'figures'
    fig_dir.mkdir(exist_ok=True, parents=True)

    with open(file_dir / 'results.pkl', 'rb') as f:
        data: List[List[LeakageBenchmarkFoldResults]] = pickle.load(f)

    # Aggregate over folds
    res = []  # type: List[LeakageBenchmarkResults]
    for dataset_res in data:
        res.append(LeakageBenchmarkResults.aggregate_fold_results(dataset_res))

    all_res = pd.concat([r.results_df for r in res])

    datasets_that_leak = all_res.loc[all_res['relative_test_loss_by_leak/GBM'] > 0, 'dataset']
    loss_for_leak_ds = all_res[all_res['dataset'].isin(datasets_that_leak)]['relative_test_loss_by_leak/GBM']
    gap_for_leak_ds = all_res[all_res['dataset'].isin(datasets_that_leak)]['misfit_gap_measure/GBM']
    print("The following datasets leak for LightGBM", list(datasets_that_leak),
          f"\nFor these datasets, the leak reduces the loss from min {loss_for_leak_ds.min():.3f}% "
          f"to max {loss_for_leak_ds.max():.3f}% (avg.: {loss_for_leak_ds.mean():.3f}%).",
          f"\nMoreover, the gap between validation and test score increases on average by {gap_for_leak_ds.mean():.3f}%.")

    # Plot leakage prevention quality (overall)
    leak_res = [r for r in res if r.dataset in list(datasets_that_leak)]
    performance_per_dataset = pd.concat(
        [r.leak_overview_df.drop(index=['all_l2_models'])[['test_score']].T.rename(index=dict(test_score=r.dataset))
         for r in res])
    cd_evaluation(performance_per_dataset, True,
                  fig_dir / 'leakage_mitigation_all_compare_cd_plot.pdf',
                  ignore_non_significance=True)

    # Plot leaks only
    cd_evaluation(performance_per_dataset[performance_per_dataset.index.isin(datasets_that_leak)], True,
                  fig_dir / 'leakage_mitigation_leak_compare_cd_plot.pdf', ignore_non_significance=True)

    # TODO: difference between distribution of all parameters for leak and non-leak datasets.
    #   (across al task types or first just for one task type)


if __name__ == '__main__':
    _run()
