from typing import Tuple, Dict
from scipy.interpolate import interp1d

import pandas as pd
import re


DEFAULT_SETUP = {('wide', 'narrow'): ['Au', 'Co', 'Ni', 'Pt'],
                 ('wide', ): ['C', 'F', 'O', 'S']}


def load_default_setup() -> Dict[str, interp1d]:

    data = {}
    for suffixes, elements in DEFAULT_SETUP.items():
        for element in elements:
            data[element] = load_multiple_files(f'data/reference_spectra/{element}', suffixes)

    return data


def load_multiple_files(root_path: str, suffixes: Tuple[str]) -> interp1d:

    dfs = [load_single_file(f'{root_path}_{suffix}.dat') for suffix in suffixes]
    df = pd.concat(dfs, ignore_index=True)

    df = df.sort_values('energy')
    df = df.drop_duplicates(subset='energy', ignore_index=True)

    return interp1d(df['energy'], df['attn_coef'], bounds_error=True)


def load_single_file(path: str) -> pd.DataFrame:

    with open(path, 'r') as f:

        header = next(f)
        density = re.search('Density=(\d*\.\d*)', header).group(1)
        density = float(density)  # in g/cm^3

    data = pd.read_csv(path, sep='  ', skiprows=2, names=['energy', 'attn_length'], engine='python')

    # Normalize to 1 g/cm^3
    data['attn_length'] *= density
    data['attn_coef'] = 1.0 / data['attn_length']
    return data
