from typing import Tuple, Dict

from scipy.interpolate import interp1d

import numpy as np
import pandas as pd
import re

from experiment_setup import PELLETS_THIRD_BATCH, NI_TRANSMISSION

ATOMIC_WEIGHTS = {
    'H': 1.008,
    'C': 12.011,
    'O': 15.999,
    'F': 18.998,
    'S': 32.06,
}

_REFERENCE_SPECTRA_FILES = {('wide', 'narrow'): ['Au', 'Co', 'Ni', 'Pt'],
                            ('wide', ): ['C', 'F', 'O', 'S']}

NICKEL_REGIMES = {
    'pre_edge': 8330.85,
    'metallic': 8333.7,
    'mixed': 8335.1,
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

    if len(experiment.detectors) == 1:
        df = pd.read_pickle(f'data/{experiment.output_path(None)}').rename(columns={'intensity': 'intensity_total'})
    else:
        detectors = experiment.detectors + ['total']
        data = {detector: pd.read_pickle(f'data/{experiment.output_path(detector)}') for detector in detectors}

        df = pd.DataFrame({f'intensity_{detector}': df['intensity'] for detector, df in data.items()})
        df['energy'] = data[1]['energy']  # Assumes all x-axes are the same

    df['energy'] *= 1000  # Convert to eV
    return df


def get_nickel_references():

    pellets = {
        'NiO': PELLETS_THIRD_BATCH[0],
        'NiSO4': PELLETS_THIRD_BATCH[1],
        'Ni-metallic': NI_TRANSMISSION,
    }

    references = {}
    for name, pellet in pellets.items():

        data = load_experiment_data(pellet)

        if pellet.name == 'Ni-metallic':

            # Data comes from transmission
            data['intensity_total'] = np.log(data['intensity_total'])

        # Subtract background
        mask_background = data['energy'] < NICKEL_REGIMES['pre_edge']
        background = data.loc[mask_background, 'intensity_total'].mean()

        data['intensity'] = data['intensity_total'] - background

        # Area normalization
        norm = np.trapz(data['intensity'], data['energy'])
        data['intensity'] /= norm
        references[pellet.name] = interp1d(data['energy'], data['intensity'], bounds_error=True)

    return references


REFERENCE_SPECTRA = load_reference_spectra()
