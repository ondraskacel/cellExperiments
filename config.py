from dataclasses import dataclass, field, InitVar
from typing import Dict, List

from scipy.interpolate import interp1d

import numpy as np

from data import ATOMIC_WEIGHTS


@dataclass
class Layer:

    depth: float  # in microns
    attn_data: Dict[str, interp1d]

    _densities: Dict[str, float] = field(init=False)

    formula: InitVar[Dict[str, int] | None] = None
    density: InitVar[float | None] = None
    densities: InitVar[Dict[str, float] | None] = None

    def __post_init__(self, formula, density, densities):

        if densities is not None:
            self._densities = densities
        else:
            assert formula is not None and density is not None

            total_atomic_weight = 0.0
            self._densities = {}
            for element, count in formula.items():
                self._densities[element] = count * ATOMIC_WEIGHTS[element]
                total_atomic_weight += self._densities[element]

            for element, count in formula.items():
                self._densities[element] = self._densities[element] * density / total_atomic_weight

    def attn_coef(self, energies: np.ndarray) -> np.ndarray:

        total_attn = np.zeros_like(energies, dtype=float)
        for element, attn in self.attn_data.items():

            if element in self._densities:
                total_attn += self._densities[element] * attn(energies)

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
