import matplotlib.pyplot as plt
import numpy as np

from experiment_setup import PELLETS_SECOND_BATCH, CELL_J, CELL_I
from materials import Layer, Cell, ANODE, NAFION
from data import load_experiment_data
from intensity import output_intensity


def run_angle_comparison_naf212_ni_au():

    theta_in = np.pi / 180 * 31
    theta_out = np.pi / 180 * np.array([41, 34, 27, 20, 13])  # Detectors C1-5

    gold_density = 19.3
    gold = Layer(depth=14.0 / gold_density * 1e-2,  # conversion cm -> m
                 densities={'Au': gold_density})

    layers = [
        ANODE,
        gold,
        NAFION,
        gold,
        NAFION,
        gold,
        ANODE,
    ]

    cell = Cell(layers)

    energies = np.array([8400])
    energy_out = 7480

    grid_size = 10000

    # Setup hypothetical densities (for angle distribution, only the relative placement matters)
    nickel = {'top': np.zeros(grid_size),
              'bottom': np.zeros(grid_size),
              'original': np.zeros(grid_size)}

    nickel['top'][:int(0.05 * grid_size)] = 1.0
    nickel['bottom'][-int(0.05 * grid_size):] = 1.0

    nickel['uniform'] = np.ones(grid_size)

    # Originally, the nickel is located near the gold
    nickel['original'][int(4 / 108 * grid_size)] = 12.0
    nickel['original'][int(54 / 108 * grid_size)] = 12.0
    nickel['original'][int(104 / 108 * grid_size)] = 11.0

    results = {name: output_intensity(theta_in, theta_out, cell, energies, energy_out, density)
               for name, density in nickel.items()}

    # real data
    cell = CELL_I[0]
    data = load_experiment_data(cell)

    # Band of energies close to 8400 ev
    results[f'real_data: {cell.output_name or cell.name}'] = data.values[777-10:777+10, :5].mean(axis=0).reshape((1, -1))

    fig, ax = plt.subplots()
    for name, result in results.items():
        ax.plot(cell.detectors, result[0, :] / result[0, 0], label=name)

    plt.legend()
    plt.show()


def run_ls_04_ccm_comparison():
    platinum_density = 21.447

    cell = Cell(layers=[
        Layer(depth=150.0 / platinum_density * 1e-2, densities={'Pt': platinum_density}),
        NAFION,
        ANODE,
    ])

    theta_in = np.pi / 180 * 31
    theta_out = np.pi / 180 * np.array([41, 34, 27, 20, 13])  # Detectors C1-5

    energies = np.array([8380])
    energy_out = 7480

    grid_size = 10000
    nickel = {'top': np.zeros(grid_size),
              'bottom': np.zeros(grid_size),
              'original': np.zeros(grid_size)}

    nickel['top'][:int(0.05 * grid_size)] = 1.0
    nickel['bottom'][-int(0.05 * grid_size):] = 1.0

    nickel['uniform'] = np.ones(grid_size)

    results = {name: output_intensity(theta_in, theta_out, cell, energies, energy_out, density)
               for name, density in nickel.items()}

    in_cell = load_experiment_data(CELL_J[0])
    ex_situ = load_experiment_data(PELLETS_SECOND_BATCH[9])

    results['in_cell'] = in_cell.values[575 - 10:575 + 10, :5].mean(axis=0).reshape((1, -1))
    results['ex_situ'] = ex_situ.values[585 - 10:585 + 10, :5].mean(axis=0).reshape((1, -1))

    fig, ax = plt.subplots()
    for name, result in results.items():
        ax.plot(CELL_J[0].detectors, result[0, :] / result[0, 0], label=name)

    plt.legend()
    plt.show()
