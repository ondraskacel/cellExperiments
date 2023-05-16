import matplotlib.pyplot as plt
import numpy as np

from experiment_setup import CELL_Q, CELL_K, CELL_L, CELL_N, CELL_P, CELL_J, PELLETS_SECOND_BATCH, CELL_R, \
    PELLETS_FIRST_BATCH
from geometry import get_geometric_factors
from modelling import fit_experiments


if __name__ == '__main__':

    cells = PELLETS_FIRST_BATCH
    coefficients = fit_experiments(cells, 1)

    first_coef = next(iter(coefficients.values()))
    reference_names = [name for name in first_coef if name != 'background']

    colors = ['red', 'green', 'blue']

    factors = get_geometric_factors()
    correction = factors['theoretical'] / factors['in_cell']

    fig, ax = plt.subplots(2, len(cells))
    coef_y_limit = 0.0
    ratio_y_limits = [np.inf, 0]

    for i, experiment in enumerate(cells):

        name = experiment.output_name or experiment.name
        detectors = experiment.detectors
        x_axis = detectors if detectors[0] is not None else [1]

        coefs_run = {}
        for j, reference in enumerate(reference_names):

            coefs_run[reference] = np.array([coefficients[(name, detector)][reference] for detector in detectors])  # * correction
            ax[0][i].scatter(x_axis, coefs_run[reference], label=reference, color=colors[j])

            coef_y_limit = max(coef_y_limit, np.max(coefs_run[reference]))

        ax[0][i].legend()
        ax[0][i].set_title(name)

        ratio = coefs_run[reference_names[0]] / coefs_run[reference_names[1]]
        ax[1][i].scatter(x_axis, ratio)
        ratio_y_limits[0] = min(ratio_y_limits[0], np.min(ratio))
        ratio_y_limits[1] = max(ratio_y_limits[1], np.max(ratio))

    for i in range(len(cells)):
        ax[0][i].set_ylim([0.0, coef_y_limit])
        ax[1][i].set_ylim(ratio_y_limits)

    plt.show()
