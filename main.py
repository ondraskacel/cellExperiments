import matplotlib.pyplot as plt
import numpy as np

from experiment_setup import CELLS_SECOND_BATCH, PELLETS_SECOND_BATCH
from data import load_experiment_data
from intensity import output_intensity
from materials import Cell, Layer, NAFION, ANODE

if __name__ == '__main__':

    platinum_density = 21.447

    cell = Cell(layers=[
        Layer(depth=150.0 / platinum_density * 1e-2, densities={'Pt': platinum_density}),
        NAFION,
        ANODE,
    ])

    theta_in = np.pi / 180 * 31
    theta_out = np.pi / 180 * np.array([41, 34, 27, 20, 13])  # Detectors C1-5

    energies = np.array([8380])

    nickel = {'top': np.zeros(10000),
              'bottom': np.zeros(10000),
              'original': np.zeros(10000)}

    nickel['top'][:500] = 1.0
    nickel['bottom'][-500:] = 1.0

    nickel['uniform'] = np.ones(10000) * 500 / 10000

    results = {name: output_intensity(1.0, theta_in, theta_out, cell, energies, 7480, density) for name, density in
               nickel.items()}

    in_cell = load_experiment_data(CELLS_SECOND_BATCH[0])
    ex_situ = load_experiment_data(PELLETS_SECOND_BATCH[9])

    results['in_cell'] = in_cell.values[575 - 10:575 + 10, :5].mean(axis=0).reshape((1, -1))
    results['ex_situ'] = ex_situ.values[585 - 10:585 + 10, :5].mean(axis=0).reshape((1, -1))

    fig, ax = plt.subplots()
    for name, result in results.items():
        ax.plot(range(1, 6), result[0, :] / result[0, 0], label=name)

    plt.legend()
    plt.show()
