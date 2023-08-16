from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from functools import partial

import pandas as pd
import numpy as np

L1_PREFIX = 'L1/OOF/'
EPS = np.finfo(np.float32).eps


@dataclass
class LeakageBenchmarkConfig:
    """
    Benchmark configuration for leakage analysis.

    :param l1_models: List of models in layer 1 to use for leakage analysis.
    :param l2_models: List of models in layer 2 to use for leakage analysis.
    :param datasets: List of datasets to use for leakage analysis.
    """

    l1_models: List[str] | None = None
    l2_models: Dict[str, List[Dict]] | None = None
    datasets: List[str] | None = None

    @property
    def default_l2(self):
        return {
            'GBM': [
                {},
                {'monotone_constraints_for_stack_features': True,
                 'monotone_constraints_method': 'advanced',
                 'monotone_penalty': 0,
                 'stack_feature_interactions_map': True,
                 'ag_args': {'name_suffix': '_mc_int'}},
                # {'only_correct_instances': True, 'ag_args': {'name_suffix': '_OCI'}},
                {'random_noise_for_stack': True, 'ag_args': {'name_suffix': '_noise_dummy'}},
                {
                    'monotone_constraints_for_stack_features': True,
                    'monotone_constraints_method': 'advanced', 'ag_args': {'name_suffix': '_monotonic'},
                    'monotone_penalty': 0,
                },
                # {'drop_duplicates': True, 'ag_args': {'name_suffix': '_dd'}},
                # {
                #     'drop_duplicates': True,
                #     'monotone_constraints_for_stack_features': True,
                #     'monotone_constraints_method': 'advanced', 'ag_args': {'name_suffix': '_monotonic_dd'},
                #     'monotone_penalty': 0,
                #
                # },
            ]
        }

    def repo_init(self, repo):
        """Init the default values for the config based on the repository."""
        self.l1_models = repo.list_models() if self.l1_models is None else self.l1_models
        self.l2_models = self.default_l2 if self.l2_models is None else self.l2_models
        self.datasets = repo.dataset_names() if self.datasets is None else self.datasets


@dataclass
class LeakageBenchmarkFoldResults:
    """
    An object to track the result of the benchmark.

    A leaderboard_df is a dataframe with three columns ['model', 'score_test', 'score_val']
    where each row corresponds to one model.


    Implicit assumption for all functions:
        - We have the score of methods. Hence, higher is better.
        - Model names are unique (also across l1 and l2 leaderboards).
    """

    fold: int
    dataset: str
    l1_leaderboard_df: pd.DataFrame
    l2_leaderboard_df: pd.DataFrame
    custom_meta_data: Dict[str, Any]
    leaderboard_df_cols = {'model', 'score_test', 'score_val'}

    is_close_func = partial(np.isclose, atol=1e-03, rtol=1e-05)

    def __post_init__(self):
        for l_b in [self.l1_leaderboard_df, self.l2_leaderboard_df]:
            assert set(l_b.columns) == self.leaderboard_df_cols, \
                f'Leaderboard columns should be {self.leaderboard_df_cols}.'

    @property
    def l2_models(self) -> List[str]:
        return sorted(list(self.l2_leaderboard_df['model']))

    @property
    def l1_models(self) -> List[str]:
        return sorted(list(self.l1_leaderboard_df['model']))

    @property
    def task_indication_columns(self) -> List[str]:
        return [c for c, _ in self._task_indicator_data]

    @staticmethod
    def print_leaderboard(leaderboard_df: pd.DataFrame):
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(leaderboard_df)

    @property
    def _task_indicator_data(self) -> List[Tuple[str, Any]]:
        return [
            ('fold', self.fold),
            ('dataset', self.dataset),
            ('eval_metric_name', self.custom_meta_data['eval_metric_name']),
            ('problem_type', self.custom_meta_data['problem_type']),
        ]

    @property
    def _basic_meta_data(self) -> List[Tuple[str, Any]]:
        return [
            ('n_features', self.custom_meta_data['n_columns']),
            ('train/n_instances', self.custom_meta_data['train_n_instances']),
            ('test/n_instances', self.custom_meta_data['test_n_instances']),
            # Duplicates - columns
            ('train/duplicates/columns/ratio', self.custom_meta_data['train_duplicated_columns']),
            ('test/duplicates/columns/ratio', self.custom_meta_data['test_duplicated_columns']),
            # Duplicates - rows
            ('train/duplicates/rows/full/ratio', self.custom_meta_data['train_l2_duplicates']),
            ('test/duplicates/rows/full/ratio', self.custom_meta_data['test_l2_duplicates']),
            ('train/duplicates/rows/feature/ratio', self.custom_meta_data['train_feature_duplicates']),
            ('test/duplicates/rows/feature/ratio', self.custom_meta_data['test_feature_duplicates']),
            ('train/duplicates/rows/feature_label/ratio', self.custom_meta_data['train_feature_label_duplicates']),
            ('test/duplicates/rows/feature_label/ratio', self.custom_meta_data['test_feature_label_duplicates']),
            # OOF Uniqueness
            ('train/unique_values_l1_models/ratio/avg', np.mean(self.custom_meta_data['train_unique_values_per_oof'])),
            ('test/unique_values_l1_models/ratio/avg', np.mean(self.custom_meta_data['test_unique_values_per_oof'])),
        ]

    @property
    def _leakage_indicator_meta_data(self) -> List[Tuple[str, Any]]:

        if self.custom_meta_data['problem_type'] != 'binary':
            return []

        return [
            # Threshold
            ('train/optimal_threshold_l1_models/avg', np.mean(
                self.custom_meta_data['optimal_threshold_train_per_oof'])),
            ('test/optimal_threshold_l1_models/avg', np.mean(
                self.custom_meta_data['optimal_threshold_test_per_oof'])),

            # Incorrect agreement
            ('train/no_model_correct_ration', self.custom_meta_data['always_wrong_row_ratio_train']),
            ('test/no_model_correct_ration', self.custom_meta_data['always_wrong_row_ratio_test']),

            # Cheat potential
            ('train/cheat_potential/duplicates_view/ratio/avg', np.mean(
                self.custom_meta_data['potential_for_cheat_stats_duplicates_view']['train_stats'][
                    'avg_potential_for_cheat_ratio'])),
            ('test/cheat_potential/duplicates_view/ratio/avg', np.mean(
                self.custom_meta_data['potential_for_cheat_stats_duplicates_view']['test_stats'][
                    'avg_potential_for_cheat_ratio'])),
            ('train/cheat_potential/tree_view/ratio/avg', np.mean(
                self.custom_meta_data['potential_for_cheat_stats_tree_view']['train_stats'][
                    'avg_potential_for_cheat_ratio'])),
            ('test/cheat_potential/tree_view/ratio/avg', np.mean(
                self.custom_meta_data['potential_for_cheat_stats_tree_view']['test_stats'][
                    'avg_potential_for_cheat_ratio'])),

            # Leaf sample size ratio
            ('train/tree_leaf_sample_size/ratio/avg', self.custom_meta_data['potential_for_cheat_stats_tree_view'][
                'train_stats']['avg_rel_sample_count']),
            ('test/tree_leaf_sample_size/ratio/avg', self.custom_meta_data['potential_for_cheat_stats_tree_view'][
                'test_stats']['avg_rel_sample_count']),
        ]

    def simulate_score(self, l2_model: str | None = None, score: str = 'test') -> float:
        """Simulate test score for the current benchmark result.

        That is, obtain the highest test performance according to validation score.
        If a l2_model name is given, the second layer is restricted to only this one l2 model.
        """
        assert score in ['test', 'validation'], f'score should be "test" or "validation".'

        l1_test_score = self.l1_leaderboard_df.iloc[self.l1_leaderboard_df['score_val'].argmax()]['score_test']
        l2_df = self.l2_leaderboard_df

        if l2_model is None:
            l2_test_score = l2_df.iloc[l2_df['score_val'].argmax()]['score_test']
            l2_score_val = l2_df['score_val'].max()
        else:
            l2_test_score = l2_df.loc[l2_df['model'] == l2_model, 'score_test'].iloc[0]
            l2_score_val = l2_df.loc[l2_df['model'] == l2_model, 'score_val'].iloc[0]

        if self.l1_leaderboard_df['score_val'].max() <= l2_score_val:
            return l2_test_score if score == 'test' else l2_score_val
        else:
            return l1_test_score if score == 'test' else self.l1_leaderboard_df['score_val'].max()

    def relative_test_score_loss_by_leak(self, l2_model: str | None = None) -> float:
        """Returns how much the leak reduced the test performance.

        Returns 0 if no performance was lost and subsequently no stack info leakage occurred.
        """

        l1_test_score = self.l1_leaderboard_df.iloc[self.l1_leaderboard_df['score_val'].argmax()]['score_test']

        if l2_model is None:
            l2_test_score = self.l2_leaderboard_df.iloc[self.l2_leaderboard_df['score_val'].argmax()]['score_test']
            l2_score_val = self.l2_leaderboard_df['score_val'].max()
        else:
            l2_test_score = self.l2_leaderboard_df.loc[self.l2_leaderboard_df['model'] == l2_model, 'score_test'].iloc[
                0]
            l2_score_val = self.l2_leaderboard_df.loc[self.l2_leaderboard_df['model'] == l2_model, 'score_val'].iloc[0]

        if self.l1_leaderboard_df['score_val'].max() <= l2_score_val:

            # As intended
            if (l1_test_score <= l2_test_score) or self.is_close_func(l1_test_score, l2_test_score):
                return 0
            else:
                # Failure case for l2 validation
                return (l1_test_score - l2_test_score) / (abs(l1_test_score) if l1_test_score != 0 else EPS)

        # - else self.l1_leaderboard_df['score_val'].max() > l2_score_val
        # Valid behavior but stacking did not work well either way.
        if l1_test_score >= l2_test_score:
            return 0
        else:
            # reverse leakage case: val score of L2 is inaccurate while test score of l2 is better than of l1;
            # resulting in picking the worse l1 model.

            # l1_test_score - l2_test_score = would show much we could have gained
            #   if val for l2 had been more accurate.
            return 0

    @property
    def leak_measures(self) -> List[Tuple[str, float]]:
        m_gbm = self.l2_misfit_gap_measure('LightGBM_BAG_L2')
        l_gbm = self.relative_test_score_loss_by_leak('LightGBM_BAG_L2')
        m_gbm_mc = self.l2_misfit_gap_measure('LightGBM_monotonic_BAG_L2')
        l_gbm_mc = self.relative_test_score_loss_by_leak('LightGBM_monotonic_BAG_L2')

        return [
            ('misfit_gap_measure/GBM', m_gbm),
            ('relative_test_loss_by_leak/GBM', l_gbm),
            ('leak_strength/GBM', m_gbm if l_gbm > 0 else 0),

            ('misfit_gap_measure/GBM-MC', m_gbm_mc),
            ('relative_test_loss_by_leak/GBM-MC', l_gbm_mc),
            ('leak_strength/GBM-MC', m_gbm_mc if l_gbm_mc > 0 else 0),
        ]

    def misfit(self, s_val, s_test):
        """How to interpret misfit:

        Overfitting occurs when misfit>0 and underfitting when misfit<0.
        For misfit=0 we do not misfit and our validation and test score align perfectly.
        """

        if self.is_close_func(s_val, s_test):
            return 0

        if s_test == 0:
            s_test = EPS

        return (s_val - s_test) / abs(s_test)

    def avg_l1_misfit(self):
        _avg = []
        l1_df = self.l1_leaderboard_df

        for l1_model in self.l1_models:
            s_val_l1 = l1_df.loc[l1_df['model'] == l1_model, 'score_val'].iloc[0]
            s_test_l1 = l1_df.loc[l1_df['model'] == l1_model, 'score_test'].iloc[0]
            _avg.append(self.misfit(s_val_l1, s_test_l1))

        return np.mean(_avg)

    def l2_misfit_gap_measure(self, l2_model: str | None = None) -> float:
        """Measure the amount of misfit per dataset based on gap between validation and test score in l2 relative to l1

        If l2_model is None, we pick the best l2 models (according to validation score) as our l2 model - simulating
        default model selection behavior.
        """
        l2_df = self.l2_leaderboard_df

        if l2_model is None:
            l2_model = l2_df.iloc[l2_df['score_val'].argmax()]['model']

        s_val_l2 = l2_df.loc[l2_df['model'] == l2_model, 'score_val'].iloc[0]
        s_test_l2 = l2_df.loc[l2_df['model'] == l2_model, 'score_test'].iloc[0]

        return self.misfit(s_val_l2, s_test_l2) - self.avg_l1_misfit()

    def get_fold_df(self) -> pd.DataFrame:
        """Aggregate all collected data to specific values per fold"""

        c_v_list = self._task_indicator_data + self._basic_meta_data \
                   + self._leakage_indicator_meta_data + self.leak_measures
        c_list = [i[0] for i in c_v_list]
        v_list = [i[1] for i in c_v_list]

        return pd.DataFrame([v_list], columns=c_list)

    def get_leak_overview_df(self) -> pd.DataFrame:
        col_name = ['l2', 'test_score', 'val_score', 'simulate_test_score', 'simulate_val_score',
                    'relative_test_score_loss_by_leak', 'leakage_misfit_gap_measure']
        full_l2_row = ['all_l2_models', np.nan, np.nan,
                       self.simulate_score(), self.simulate_score(score='validation'),
                       self.relative_test_score_loss_by_leak(), self.l2_misfit_gap_measure()]

        res = [full_l2_row]
        for l2_model in self.l2_models:
            row = [
                l2_model,
                self.l2_leaderboard_df.loc[self.l2_leaderboard_df['model'] == l2_model, 'score_test'].iloc[0],
                self.l2_leaderboard_df.loc[self.l2_leaderboard_df['model'] == l2_model, 'score_val'].iloc[0],
                self.simulate_score(l2_model),
                self.simulate_score(l2_model, score='validation'),
                self.relative_test_score_loss_by_leak(l2_model),
                self.l2_misfit_gap_measure(l2_model)
            ]

            res.append(row)

        return pd.DataFrame(res, columns=col_name)


@dataclass
class LeakageBenchmarkResults:
    dataset: str
    eval_metric_name: str
    problem_type: str
    l1_models: List[str]
    l2_models: List[str]
    results_df: pd.DataFrame
    leak_overview_df: pd.DataFrame

    @staticmethod
    def aggregate_fold_results(fold_results: List[LeakageBenchmarkFoldResults]) -> LeakageBenchmarkResults:
        consistency_template = fold_results[0]
        task_indicator_data = consistency_template._task_indicator_data
        l1_models = consistency_template.l1_models
        l2_models = consistency_template.l2_models

        # Check consistency of all fold results
        for fold_result in fold_results:
            assert task_indicator_data == fold_result._task_indicator_data
            assert l1_models == fold_result.l1_models
            assert l2_models == fold_result.l2_models

        # Aggregate all fold results
        results_df, leak_overview_df = [], []

        for fold_result in fold_results:
            results_df.append(fold_result.get_fold_df())
            leak_overview_df.append(fold_result.get_leak_overview_df())

        results_df = pd.concat(results_df).groupby(
            list(set(consistency_template.task_indication_columns) - {'fold', })).mean()
        results_df = results_df.reset_index().drop(columns=['fold'])
        results_df['folds'] = len(fold_results)

        leak_overview_df = pd.concat(leak_overview_df).groupby(by=['l2']).mean()

        task_indicator_data_dict = {k: v for k, v in task_indicator_data}

        return LeakageBenchmarkResults(
            dataset=task_indicator_data_dict['dataset'],
            eval_metric_name=task_indicator_data_dict['eval_metric_name'],
            problem_type=task_indicator_data_dict['problem_type'],
            l1_models=l1_models,
            l2_models=l2_models,
            results_df=results_df,
            leak_overview_df=leak_overview_df
        )
