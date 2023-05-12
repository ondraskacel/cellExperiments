import matplotlib.pyplot as plt
import numpy as np

from config import Layer, Cell
from data import load_default_setup, load_experiment_data
from intensity import output_intensity


if __name__ == '__main__':

    theta_in = np.pi / 6
    theta_out = np.pi / 180 * np.array([41, 34, 27, 20, 13])  # Detectors C1-5

    attn_data = load_default_setup()

    anode = Layer(depth=4.0,
                  attn_data=attn_data,
                  densities={'Pt': 0.1975, 'C': 0.960, 'F': 0.421, 'O': 0.048, 'S': 0.019})

    gold_density = 19.3
    gold = Layer(depth=14.0 / gold_density * 1e-2,  # conversion cm -> m
                 attn_data=attn_data,
                 densities={'Au': gold_density})

    nafion = Layer(depth=50.0,
                   attn_data=attn_data,
                   formula={'H': 1, 'C': 9, 'O': 5, 'F': 17, 'S': 1},
                   density=1.9)

    layers = [
        anode,
        # gold,
        nafion,
        # gold,
        nafion,
        # gold,
        anode,
    ]

    cell = Cell(layers)

    energies = np.array([8400])

    nickel = {}

    nickel['top'] = np.zeros(10000)
    nickel['top'][:500] = 1.0

    nickel['bottom'] = np.zeros(10000)
    nickel['bottom'][-500:] = 1.0

    nickel['uniform'] = np.ones(10000) * 500 / 10000

    nickel['original'] = np.ones(10000)
    nickel['original'][int(4 / 108) * 10000] = 12.0
    nickel['original'][int(54 / 108) * 10000] = 12.0
    nickel['original'][int(104 / 108) * 10000] = 11.0

    results = {name: output_intensity(1.0, theta_in, theta_out, cell, energies, 7480, density) for name, density in nickel.items()}

    # real data
    experiment = 'Ref-CCM-Naf212-Ni-Au-I_a'
    suffix = '_new_y_1'

    data = load_experiment_data(experiment, suffix)
    results['real'] = data.values[777-10:777+10, :5].mean(axis=0).reshape((1, -1))  # closest to 8400 ev

    fig, ax = plt.subplots()
    for name, result in results.items():
        ax.scatter(range(1, 6), result[0, :] / result[0, 0], label=name)

    plt.legend()
    plt.show()
