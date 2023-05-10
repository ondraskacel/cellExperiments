from typing import Union

from scipy.optimize import minimize

import matplotlib.pyplot as plt
import numpy as np

from config import Layer, Cell
from data import load_default_setup


def simple_integration(x: np.ndarray, low=0.0, high=1.0) -> Union[float, np.ndarray]:
    """
    Performs a simple '(1 + 2 + 2 + ... + 2 + 1) / norm' integration over the first axis of x
    """

    assert len(x) > 2

    norm = len(x) - 1
    total = np.sum(x, axis=0) - 0.5 * (x[0] + x[-1])
    return total / norm * (high - low)


def output_intensity(
        input_intensity: float,
        theta_in: float,
        theta_out: np.ndarray,
        cell: Cell,
        energies_in: np.ndarray,
        energy_out: float,
        nickel_density: np.ndarray,
        ) -> Union[float, np.ndarray]:
    """
    Assumes:
    - nickel does not contribute to the overall absorption
    - the outgoing angle is a single number
    TODO: assess (and remove?) the assumptions
    """

    # The sine comes from the change of variables length -> depth
    multiplier = input_intensity / np.sin(theta_in)

    grid_size = len(nickel_density)
    depth_grid = np.linspace(0.0, cell.total_depth, grid_size)

    decay_in = np.exp(-cell.log_decay(energies_in, depth_grid) / np.sin(theta_in))
    decay_out = np.exp(-cell.log_decay(np.array([energy_out]), depth_grid) / np.sin(theta_out))

    # axis 0 <-> depth; axis 1 <-> energy_in; axis 2 <-> theta_out
    integrand = nickel_density[:, None, None] * decay_in[:, :, None] * decay_out[:, None, :]
    return simple_integration(multiplier * integrand, high=cell.total_depth)


def get_density(beta: np.ndarray):

    x_grid = np.linspace(0.0, 1.0, 51)
    # return beta[0] * np.exp(-np.square(beta[1] - x_grid) / np.square(beta[2]))  # Gaussian parametrization
    log_density = beta[0]
    log_density += beta[1] * (2 * x_grid - 1)
    log_density += beta[2] * (6 * np.square(x_grid) - 6 * x_grid + 1)
    return np.exp(log_density)


if __name__ == '__main__':

    theta_in = np.pi / 6
    theta_out = np.pi / 180 * np.array([15, 25, 35, 45, 55])

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

    # depth = 10.0
    # absorption_coef = 0.1
    #
    # actual_intensity = output_intensity(1.0, theta_in, theta_out, absorption_coef, density_0, depth)
    # sigma = np.min(actual_intensity) * 0.1
    #
    # def estimated_output(beta):
    #
    #     density = get_density(beta)
    #     return output_intensity(1.0, theta_in, theta_out, absorption_coef, density, depth)
    #
    #
    # def loss(beta, actual):
    #
    #     est = estimated_output(beta)
    #     return np.sum(np.square(actual - est))
    #
    # fig, ax = plt.subplots()
    # for i in range(10):
    #
    #     measured_intensity = actual_intensity + sigma * np.random.normal(size=len(actual_intensity))
    #
    #     best_loss = np.inf
    #     best_params = None
    #
    #     for j in range(10):
    #         res = minimize(lambda x: loss(x, measured_intensity),
    #                        x0=np.random.normal(size=3),
    #                        bounds=[[None, None], [None, None], [None, 0.0]],
    #                        method='L-BFGS-B', jac='2-point')
    #
    #         if res['fun'] < best_loss:
    #             best_loss = res['fun']
    #             best_params = res['x']
    #
    #     ax.plot(get_density(best_params), label=str(i))
    #
    # plt.legend()
    # plt.show()
