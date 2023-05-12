import matplotlib.pyplot as plt
import numpy as np

from config import Layer, Cell
from data import load_default_setup
from intensity import output_intensity


def get_density(beta: np.ndarray):

    x_grid = np.linspace(0.0, 1.0, 51)

    log_density = beta[0]
    log_density += beta[1] * (2 * x_grid - 1)
    log_density += beta[2] * (6 * np.square(x_grid) - 6 * x_grid + 1)

    return np.exp(log_density)


if __name__ == '__main__':

    theta_in = np.pi / 6
    theta_out = np.pi / 180 * np.array([41, 34, 27, 20, 13])  # Detectors C1-5

    attn_data = load_default_setup()

    anode = Layer(densities={'Pt': 0.1975, 'C': 0.960, 'F': 0.421, 'O': 0.048, 'S': 0.019},
                  depth=4.0,
                  attn_data=attn_data)

    layers = [
        anode,
        Layer(densities={'Au': 20.0}, depth=0.02, attn_data=attn_data),
        anode,
    ]

    cell = Cell(layers)

    energies = np.linspace(7400.0, 13000.0, 1000)
    nickel_density = np.linspace(0.0, 1.0, 25)
    results = output_intensity(1.0, theta_in, theta_out, cell, energies, 7480, nickel_density)

    for i, theta_out in enumerate(theta_out):
        plt.plot(energies, results[:, i], label=f'{180 / np.pi * theta_out:.1f} deg')

    plt.legend()
    plt.show()
