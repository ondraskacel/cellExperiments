from typing import Union

import numpy as np

from materials import Cell


def output_intensity(
        theta_in: float,
        theta_out: np.ndarray,
        cell: Cell,
        energies_in: np.ndarray,
        energy_out: float,
        nickel_density: np.ndarray,
        ) -> Union[float, np.ndarray]:
    """
    The function ignores multiplicative factors like:
        - input intensity
        - intensity emitted per intensity incoming per density
        ...
    """

    # The sine comes from the change of variables length -> depth
    multiplier = 1.0 / np.sin(theta_in)

    grid_size = len(nickel_density)
    depth_grid = np.linspace(0.0, cell.total_depth, grid_size)

    decay_in = np.exp(-cell.log_decay(energies_in, depth_grid) / np.sin(theta_in))
    decay_out = np.exp(-cell.log_decay(np.array([energy_out]), depth_grid) / np.sin(theta_out))

    # axis 0 <-> depth; axis 1 <-> energy_in; axis 2 <-> theta_out
    integrand = nickel_density[:, None, None] * decay_in[:, :, None] * decay_out[:, None, :]
    return np.trapz(multiplier * integrand, depth_grid, axis=0)
