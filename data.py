from typing import Tuple, Dict
from scipy.interpolate import interp1d

import pandas as pd
import re


ATOMIC_WEIGHTS = {
    'H': 1.008,
    'C': 12.011,
    'O': 15.999,
    'F': 18.998,
    'S': 32.06,
}

_REFERENCE_SPECTRA_FILES = {('wide', 'narrow'): ['Au', 'Co', 'Ni', 'Pt'],
                            ('wide', ): ['C', 'F', 'O', 'S']}


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
        return pd.read_pickle(f'data/{experiment.output_path(None)}')

    data = {detector: pd.read_pickle(f'data/{experiment.output_path(detector)}') for detector in experiment.detectors}

    df = pd.DataFrame({f'intensity_{detector}': df['intensity'] for detector, df in data.items()})
    df['energy'] = data[1]['energy']  # Assumes all x-axes are the same

    return df


REFERENCE_SPECTRA = load_reference_spectra()


if __name__ == '__main__':

    from experiment_setup import CELLS_FIRST_BATCH, PELLETS_FIRST_BATCH, PELLETS_SECOND_BATCH

    data = {}
    for cell in CELLS_FIRST_BATCH:
        data[(cell.name, cell.output_suffix)] = load_experiment_data(cell)

    for pellet in PELLETS_FIRST_BATCH + PELLETS_SECOND_BATCH:
        data[(pellet.name, pellet.output_suffix)] = load_experiment_data(pellet)

    import matplotlib.pyplot as plt

    for name, df in data.items():
        plt.plot(df.filter(regex='intensity'), label=f'{name[0]}{name[1]}')

    plt.legend()
    plt.show()
