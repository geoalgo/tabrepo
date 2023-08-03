from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

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
            'GBM':
                [
                    {},
                    {
                        'monotone_constraints_for_stack_features': True,
                        'monotone_constraints_method': 'basic', 'ag_args': {'name_suffix': '_monotonic'}
                    },
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
    leaderboard_df_cols = {'model', 'score_test', 'score_val'}

    def __post_init__(self):
        for l_b in [self.l1_leaderboard_df, self.l2_leaderboard_df]:
            assert set(l_b.columns) == self.leaderboard_df_cols, \
                f'Leaderboard columns should be {self.leaderboard_df_cols}.'

    @staticmethod
    def print_leaderboard(leaderboard_df: pd.DataFrame):
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(leaderboard_df)
