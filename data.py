from typing import Tuple, Dict

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import numpy as np
import pandas as pd
import re

from experiment_setup import PELLETS_THIRD_BATCH, NI_TRANSMISSION, NI_FOIL, PELLETS_FIRST_BATCH, PELLETS_SECOND_BATCH

ATOMIC_WEIGHTS = {
    'H': 1.008,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'F': 18.998,
    'S': 32.06,
}

_REFERENCE_SPECTRA_FILES = {('wide', 'narrow'): ['Au', 'Co', 'Ni', 'Pt'],
                            ('wide', ): ['H', 'C', 'F', 'N', 'O', 'S']}

NICKEL_REGIMES = {
    'pre_edge': 8330.85,
    'fit_from': 8329,
    'fit_to': 8347,
}


def load_reference_spectra() -> Dict[str, interp1d]:

    data = {}
    for suffixes, elements in _REFERENCE_SPECTRA_FILES.items():
        for element in elements:
            data[element] = load_multiple_references(f'data/reference_spectra/{element}', suffixes)

    return data


def load_multiple_references(root_path: str, suffixes: Tuple[str]) -> interp1d:

    dfs = [load_single_reference(f'{root_path}_{suffix}.dat') for suffix in suffixes]
    df = pd.concat(dfs, ignore_index=True)

    df = df.sort_values('energy')
    df = df.drop_duplicates(subset='energy', ignore_index=True)

    return interp1d(df['energy'], df['attn_coef'], bounds_error=True)


def load_single_reference(path: str) -> pd.DataFrame:

    with open(path, 'r') as f:

        header = next(f)
        density = re.search('Density=(\d*\.\d*)', header).group(1)
        density = float(density)  # in g/cm^3

    data = pd.read_csv(path, sep='  ', skiprows=2, names=['energy', 'attn_length'], engine='python')

    # Normalize to 1 g/cm^3
    data['attn_length'] *= density
    data['attn_coef'] = 1.0 / data['attn_length']
    return data


def load_experiment_data(experiment):

    data = {detector: pd.read_pickle(f'data/{experiment.output_path(detector)}') for detector in experiment.detectors}
    first_detector = experiment.detectors[0]

    if len(experiment.detectors) == 1:
        df = data[first_detector].rename(columns={'intensity': 'intensity_total'})
    else:
        data['total'] = pd.read_pickle(f'data/{experiment.output_path("total")}')

        df = pd.DataFrame({f'intensity_{detector}': df['intensity'] for detector, df in data.items()})
        df['energy'] = data[first_detector]['energy']  # Assumes all x-axes are the same

    df['energy'] *= 1000  # Convert to eV
    return df


def get_nickel_references(plot=False):

    experiments = {
        # 'NiO': PELLETS_THIRD_BATCH[0],
        'NiSO4': PELLETS_THIRD_BATCH[1],
        # 'NiAc2': PELLETS_FIRST_BATCH[5],
        # 'NiAcAc2': PELLETS_FIRST_BATCH[6],
        # 'NiOH2': PELLETS_FIRST_BATCH[7],
        'PtNi-dealloyed': PELLETS_SECOND_BATCH[0],
        # 'Ni-metallic_transmission': NI_TRANSMISSION,
        # 'Ni-metallic_foil': NI_FOIL[1],
    }

    data = {name: load_experiment_data(experiment) for name, experiment in experiments.items()}

    # Get common energy range
    energy_range = [max((df['energy'].min() for df in data.values())),
                    min((df['energy'].max() for df in data.values()))]

    references = {}
    for name, df in data.items():

        mask = (df['energy'] > energy_range[0]) & (df['energy'] < energy_range[1])
        df = df[mask].reset_index(drop=True)

        energy = df['energy']
        intensity = df['intensity_total']

        if name == 'Ni-metallic_transmission':

            # Data comes from transmission
            intensity = np.log(intensity)

        # Subtract background
        mask_background = energy < NICKEL_REGIMES['pre_edge']
        background = intensity.loc[mask_background].mean()

        intensity = intensity - background

        # Area normalization
        norm = np.trapz(intensity, energy)
        intensity /= norm

        references[name] = interp1d(energy, intensity, bounds_error=True)

    if plot:
        energy_grid = np.linspace(energy_range[0] + 1, energy_range[1] - 1, 1000)
        for name, intensity in references.items():
            plt.plot(energy_grid, intensity(energy_grid), label=name)

        for name, value in NICKEL_REGIMES.items():
            plt.axvline(value, label=name)

        plt.legend()
        plt.show()

    return references


REFERENCE_SPECTRA = load_reference_spectra()
