from dataclasses import dataclass, field, InitVar
from typing import Dict, List

import numpy as np

from data import ATOMIC_WEIGHTS, REFERENCE_SPECTRA


@dataclass
class Layer:

    depth: float  # in microns

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

        for element, density in self._densities.items():

            if element not in REFERENCE_SPECTRA:
                print(f'Warning: {element} not in reference spectra')
            else:
                total_attn += self._densities[element] * REFERENCE_SPECTRA[element](energies)

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


ANODE = Layer(depth=4.0, densities={'Pt': 0.1975, 'C': 0.960, 'F': 0.421, 'O': 0.048, 'S': 0.019})
CATHODE = Layer(depth=10.0, densities={'Pt': 0.130, 'C': 0.509, 'F': 0.233, 'O': 0.027, 'S': 0.011, 'Ni': 0.017})
GDL = Layer(depth=215.0, densities={'C': 0.326})
KAPTON = Layer(depth=65.0, formula={'H': 10, 'C': 22, 'O': 5, 'N': 2}, density=1.42)
NAFION = Layer(depth=50.0, formula={'H': 1, 'C': 9, 'O': 5, 'F': 17, 'S': 1}, density=1.96)
