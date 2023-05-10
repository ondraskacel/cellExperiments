from dataclasses import dataclass, field
from typing import Dict, List

from scipy.interpolate import interp1d

import numpy as np


@dataclass
class Layer:

    densities: Dict[str, float]
    depth: float  # in microns
    attn_data: Dict[str, interp1d]

    def attn_coef(self, energies: np.ndarray) -> np.ndarray:

        total_attn = np.zeros_like(energies, dtype=float)
        for element, attn in self.attn_data.items():

            if element in self.densities:
                total_attn += self.densities[element] * attn(energies)

        return total_attn


@dataclass
class Cell:

    layers: List[Layer]
    layer_depths: np.ndarray = field(init=False)
    total_depth: float = field(init=False)

    def __post_init__(self):
        self.layer_depths = np.array([layer.depth for layer in self.layers])
        self.total_depth = self.layer_depths.sum()

    def log_decay(self, energies: np.ndarray, depths: np.ndarray):

        attn_coefs = np.array([layer.attn_coef(energies) for layer in self.layers])
        bounds = np.cumsum(self.layer_depths) - self.layer_depths

        # axis 0 <-> desired_depths, axis 1 <-> weights
        weights = np.clip(depths[:, None] - bounds[None, :], 0.0, self.layer_depths[None, :])
        return weights @ attn_coefs


if __name__ == '__main__':

    from data import load_default_setup
    data = load_default_setup()

    layers = [
        Layer(densities={'Au': 1.0}, depth=5.0, attn_data=data),
        Layer(densities={'Pt': 1.0}, depth=2.0, attn_data=data),
        Layer(densities={'Ni': 10.0, 'Au': 1.0}, depth=3.0, attn_data=data),
    ]

    cell = Cell(layers)

    energies = np.linspace(data['Au'].x.min(), data['Au'].x.max(), 11)
    depth_grid = np.linspace(0.0, 10.0, 101)
    log_decay = cell.log_decay(energies, depth_grid)

    import matplotlib.pyplot as plt

    for i, energy in enumerate(energies):
        plt.plot(depth_grid, log_decay[:, i], label=str(energy))

    plt.legend()
    plt.show()