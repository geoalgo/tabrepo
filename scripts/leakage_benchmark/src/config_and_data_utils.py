from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd

L1_PREFIX = 'L1/OOF/'


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

    """
    fold: int
    dataset: str
    l1_leaderboard_df: pd.DataFrame
    l2_leaderboard_df: pd.DataFrame
    custom_meta_data: Dict[str, Any]
    leaderboard_df_cols = {'model', 'score_test', 'score_val'}

    def __post_init__(self):
        for l_b in [self.l1_leaderboard_df, self.l2_leaderboard_df]:
            assert set(l_b.columns) == self.leaderboard_df_cols, \
                f'Leaderboard columns should be {self.leaderboard_df_cols}.'

    @staticmethod
    def print_leaderboard(leaderboard_df: pd.DataFrame):
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(leaderboard_df)

    # def rank(self):
    #
    #     return

    def model_selection_l1_l2_test_score_difference(self, l2_model=None, noise_level=0.005) -> float:
        # Returns how much better l1 is than l2 (if it is better, otherwise 0)
        l1_test_score = self.l1_leaderboard_df.iloc[self.l1_leaderboard_df['score_val'].argmax()]['score_test']

        if l2_model is None:
            l2_test_score = self.l2_leaderboard_df.iloc[self.l2_leaderboard_df['score_val'].argmax()]['score_test']
            l2_score_val = self.l2_leaderboard_df['score_val'].max()
        else:
            l2_test_score = self.l2_leaderboard_df.loc[self.l2_leaderboard_df['model'] == l2_model, 'score_test'].iloc[0]
            l2_score_val = self.l2_leaderboard_df.loc[self.l2_leaderboard_df['model'] == l2_model, 'score_val'].iloc[0]

        # Rank test
        # if self.l1_leaderboard_df['score_val'].max() <= l2_score_val:
        #     return l2_test_score
        # else:
        #     return l1_test_score

        # noise likely, ignore
        if abs(l1_test_score - l2_test_score) <= noise_level:
            return 0

        # as intended
        if (self.l1_leaderboard_df['score_val'].max() <= l2_score_val) and \
                (l1_test_score <= l2_test_score):
            return -3

        # Valid behavior but stacking did not work well either way.
        if (self.l1_leaderboard_df['score_val'].max() >= l2_score_val) and \
                (l1_test_score >= l2_test_score):
            return -2

        # reverse leakage case
        # (val score of L2 is very bad while its test score is much better, resulting in us picking the worse l1 model)
        if (self.l1_leaderboard_df['score_val'].max() >= l2_score_val) and \
                (l1_test_score <= l2_test_score):
            return l1_test_score - l2_test_score

        # leakage case
        if (self.l1_leaderboard_df['score_val'].max() <= l2_score_val) and \
                (l1_test_score >= l2_test_score):
            return l1_test_score - l2_test_score

        # failure case
        return float('nan')
