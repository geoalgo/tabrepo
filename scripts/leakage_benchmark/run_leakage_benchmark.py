from __future__ import annotations

from autogluon_zeroshot.utils.cache import cache_function
from autogluon_zeroshot.repository import EvaluationRepository, EvaluationRepositoryZeroshot
from scripts.leakage_benchmark.src.config_and_data_utils import LeakageBenchmarkConfig

from scripts.leakage_benchmark.src.stacking_simulator import obtain_input_data_for_l2, autogluon_l2_runner


def _leakage_analysis(repo, lbc, dataset, fold):
    print(f'Leakage Analysis for {dataset}, fold {fold}...')
    l2_X_train, y_train, l2_X_test, y_test, eval_metric, oof_col_names = \
        obtain_input_data_for_l2(repo, lbc.l1_models, dataset, fold)
    autogluon_l2_runner(l2_X_train, y_train, l2_X_test, y_test, eval_metric, oof_col_names)
    print('... done.')


def analyze_starter(repo: EvaluationRepositoryZeroshot, lbc: LeakageBenchmarkConfig):
    # Init
    lbc.repo_init(repo)

    # Stats
    n_datasets = len(lbc.datasets)
    print(f'n_l1_models={len(lbc.l1_models)} | l1_models={lbc.l1_models}')
    print(f'n_l2_models={len(lbc.l2_models)} | l2_models={lbc.l2_models}')
    print(f'n_datasets={n_datasets}')

    # Loop over datasets for benchmark
    for dataset_num, dataset in enumerate(lbc.datasets):
        for fold in repo.folds:
            _leakage_analysis(repo, lbc, dataset, fold)


if __name__ == '__main__':
    # Download repository from S3 and cache it locally for re-use in future calls
    repository: EvaluationRepositoryZeroshot = cache_function(
        fun=lambda: EvaluationRepository.load('s3://autogluon-zeroshot/repository/BAG_D244_F1_C16_micro.pkl'),
        cache_name="repo_micro",
    ).to_zeroshot()
    init_lbc = LeakageBenchmarkConfig(
        l1_models=['RandomForest_c1_BAG_L1'],
        datasets=['airlines']
    )
    analyze_starter(repo=repository, lbc=init_lbc)
