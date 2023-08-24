from collections import Counter
from typing import List
import logging

import numpy as np

from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def caruana_weighted(predictions: List[np.ndarray], labels: np.ndarray, seed,
                     n_iterations, loss_function):
    """Caruana's ensemble selection with replacement."""

    # -- Init Vars
    num_input_models_ = len(predictions)
    ensemble = []  # type: List[np.ndarray]
    trajectory = []  # contains iteration best
    val_loss_over_iterations_ = []  # contains overall best
    order = []
    rand = check_random_state(seed)
    weighted_ensemble_prediction = np.zeros(predictions[0].shape, dtype=np.float64)

    fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape, dtype=np.float64)

    for i in range(n_iterations):
        logger.debug(f"Iteration {i}")

        ens_size = len(ensemble)
        if ens_size > 0:
            np.add(weighted_ensemble_prediction, ensemble[-1], out=weighted_ensemble_prediction)

        # -- Process Iteration Solutions
        losses = np.zeros((len(predictions)), dtype=np.float64)

        for j, pred in enumerate(predictions):
            np.add(weighted_ensemble_prediction, pred, out=fant_ensemble_prediction)
            np.multiply(fant_ensemble_prediction, (1. / float(ens_size + 1)), out=fant_ensemble_prediction)
            losses[j] = loss_function(labels, fant_ensemble_prediction)

        # -- Eval Iteration results
        all_best = np.argwhere(losses == np.nanmin(losses)).flatten()
        best = rand.choice(all_best)  # break ties randomly
        ensemble_loss = losses[best]

        ensemble.append(predictions[best])
        trajectory.append(ensemble_loss)
        order.append(best)

        # Build Correct Validation loss list
        if not val_loss_over_iterations_:
            # Init
            val_loss_over_iterations_.append(ensemble_loss)
        elif val_loss_over_iterations_[-1] > ensemble_loss:
            # Improved
            val_loss_over_iterations_.append(ensemble_loss)
        else:
            # Not Improved
            val_loss_over_iterations_.append(val_loss_over_iterations_[-1])

        # -- Break for special cases
        #   - If we only have a pool of base models of size 1 (code found the single best model)
        #   - If we find a perfect ensemble/model, stop early
        if (len(predictions) == 1) or (ensemble_loss == 0):
            break

    indices_ = order
    trajectory_ = trajectory
    min_loss = np.min(trajectory_)
    idx_best = trajectory_.index(min_loss)
    indices_ = indices_[:idx_best + 1]
    # trajectory_ = trajectory_[:idx_best + 1]
    n_iterations = idx_best + 1
    # validation_loss_ = trajectory_[idx_best]

    ensemble_members = Counter(indices_).most_common()
    weights = np.zeros((num_input_models_,), dtype=np.float64)

    for ensemble_member in ensemble_members:
        weight = float(ensemble_member[1]) / n_iterations
        weights[ensemble_member[0]] = weight

    if np.sum(weights) < 1:
        weights = weights / np.sum(weights)

    return weights, val_loss_over_iterations_


def roc_auc_binary_loss_proba(y_true, y_pred_proba):
    return 1 - roc_auc_score(y_true, y_pred_proba)  # [:, 1]
