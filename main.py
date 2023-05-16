import matplotlib.pyplot as plt
import numpy as np

from experiment_setup import CELL_Q, CELL_K, CELL_L, CELL_N, CELL_P, CELL_J, PELLETS_SECOND_BATCH, CELL_R, \
    PELLETS_FIRST_BATCH
from geometry import get_geometric_factors
from modelling import fit_experiments


def plot_single_experiment(experiment, coefs, ax, geometric_correction):

    name = experiment.output_name or experiment.name  # (Possibly) a shorter name
    detectors = experiment.detectors
    reference_names = [name for name in coefs[detectors[0]] if name != 'background']

    # Compute geometric correction
    if geometric_correction is not None:
        factors = get_geometric_factors()
        correction = factors['theoretical'] / factors[geometric_correction]

    coefs_run = {}  # Rearranged coefficients by reference and detector
    for j, reference in enumerate(reference_names):

        params = np.array([coefs[detector][reference] for detector in detectors])
        if geometric_correction is not None:
            params *= correction  # noqa

        ax[0].scatter(detectors, params, label=reference, color='rgb'[j % 3])
        coefs_run[reference] = params

    ax[0].legend()
    ax[0].set_title(name)

    ratio = coefs_run[reference_names[0]] / coefs_run[reference_names[1]]
    ax[1].scatter(detectors, ratio)


def plot_experiments(experiments, coefficients, geometric_correction=None):

    # Setup plots
    fig, ax = plt.subplots(2, len(experiments))

    for i, experiment in enumerate(experiments):
        coefs = coefficients[f'{experiment.name}{experiment.output_name}']
        plot_single_experiment(experiment, coefs, [ax[0][i], ax[1][i]], geometric_correction)

    # Compute common axes
    coef_y_limit = max(ax_.get_ylim()[1] for ax_ in ax[0])
    ratio_y_limits = min(ax_.get_ylim()[0] for ax_ in ax[1]), max(ax_.get_ylim()[1] for ax_ in ax[1])

    for i in range(len(experiments)):
        ax[0][i].set_ylim([0.0, coef_y_limit])
        ax[1][i].set_ylim(ratio_y_limits)

    plt.show()


if __name__ == '__main__':

    cells_ = CELL_R[:3] + CELL_R[9:]
    fit = fit_experiments(cells_, -1)

    plot_experiments(cells_, fit)

