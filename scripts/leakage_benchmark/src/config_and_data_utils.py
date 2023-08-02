from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

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
            'RF': [{'criterion': 'gini'}],
            'CAT': [{}],
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
    TODO: start using this. add print leaderboard function here.

    Tracking the result of the benchmark
    """
    overfitting_score: float
    leaderboard_df : pd.DataFrame
