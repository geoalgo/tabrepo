from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class LeakageBenchmarkConfig:
    """
    Benchmark configuration for leakage analysis.

    :param l1_models: List of models in layer 1 to use for leakage analysis.
    :param l2_models: List of models in layer 2 to use for leakage analysis.
    :param datasets: List of datasets to use for leakage analysis.
    """

    l1_models: List[str] | None = None
    l2_models: List[str] | None = None
    datasets: List[str] | None = None

    def repo_init(self, repo):
        """Init the default values for the config based on the repository."""
        self.l1_models = repo.list_models() if self.l1_models is None else self.l1_models
        self.l2_models = ['RF', 'LGB', 'LR'] if self.l2_models is None else self.l2_models
        self.datasets = repo.dataset_names() if self.datasets is None else self.datasets


@dataclass
class LeakageBenchmarkFoldResults:
    """
    Tracking the result of the benchmark
    """
    overfitting_score: float
