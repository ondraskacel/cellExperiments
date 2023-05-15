import matplotlib.pyplot as plt
import numpy as np

from experiment_setup import CELLS_SECOND_BATCH
from data import load_experiment_data, get_nickel_references
from modelling import fit_nickel_spectra


if __name__ == '__main__':

    data = {f'{cell.name}{cell.output_suffix[1:]}': load_experiment_data(cell) for cell in CELLS_SECOND_BATCH[14:18]}

    references = get_nickel_references()
    names = ['NiSO4', 'PtNi-dealloyed']

    detectors = [1, 2, 3, 4, 5]

    coefficients = {}
    for i, (run, df) in enumerate(data.items()):

        mask_energy = (df['energy'] < 8347) & (df['energy'] > 8329)

        for detector in detectors:
            coefficients[(run, detector)] = fit_nickel_spectra(df.loc[mask_energy].reset_index(drop=True),
                                                               references, names, detector)

    plt.show()
    fig, ax = plt.subplots(2, len(data))
    colors = ['red', 'green', 'blue', 'black']

    for i, (run, df) in enumerate(data.items()):

        coefs = {}
        for j, name in enumerate(names):
            coefs[name] = np.array([coefficients[(run, detector)][name] for detector in detectors])

            ax[0][i].scatter(detectors, coefs[name], label=name, color=colors[j])
        ax[0][i].legend()
        ax[0][i].set_title(run)
        # ax[0][i].set_ylim([0.0, 0.7])

        ax[1][i].plot(detectors, coefs['NiSO4'] / coefs['PtNi-dealloyed'])
        # ax[1][i].set_ylim([10.0, 21.0])

    plt.show()
