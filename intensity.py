from typing import Union

import numpy as np

from materials import Cell


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
