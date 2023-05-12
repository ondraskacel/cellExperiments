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


def load_pellet_data(pellet):

    return pd.read_pickle(f'data/pellet_data/{pellet}.pickle')


def load_experiment_data(experiment, suffix):

    data = {detector: pd.read_pickle(f'data/experiment_data/{experiment}_{detector}{suffix}.pickle') for detector in range(1, 6)}

    df = pd.DataFrame({f'intensity_{detector}': df['intensity'] for detector, df in data.items()})
    df['energy'] = data[1]['energy']  # Assumes all x-axes are the same

    return df


if __name__ == '__main__':

    names = 'FGHIIII'
    experiments = [f'Ref-CCM-Naf212-Ni-Au-{c}_a' for c in names]
    suffixes = ['', '', '', '_hs', '_no_hs', '_new_y_1', '_new_y_2']

    data = {f'{c}{suffix}': load_experiment_data(experiment, suffix) for c, experiment, suffix in zip(names, experiments, suffixes)}

    import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(7)
    # for i, k in enumerate(data.keys()):
    #     data[k].filter(regex='intensity').plot(ax=ax[i])
    #
    # plt.show()

    # areas = {k: df.filter(regex='intensity').sum(axis=0) for k, df in data.items()}
    #
    # fig, ax = plt.subplots()
    # for c, area in areas.items():
    #     ax.plot(area, label=c)
    #
    # ax.legend()
    # plt.show()

    # fig, ax = plt.subplots()
    # for detector in range(1, 6):
    #     plt.plot(data['I_new_y_1']['energy'], data['I_new_y_1'][f'intensity_{detector}'], label=f'new_y_1_{detector}')
    #     plt.plot(data['I_new_y_2']['energy'], data['I_new_y_2'][f'intensity_{detector}'], label=f'new_y_2_{detector}')
    #
    # plt.legend()
    # plt.show()

    experiment = 'F'

    fig, ax = plt.subplots(2)
    for detector in range(1, 6):

        df = data[experiment]
        ax[0].plot(df['energy'], df[f'intensity_{detector}'], label=f'{experiment}_{detector}')

        if detector < 5:
            ax[1].plot(df['energy'],
                       df[f'intensity_{detector}'] / df[f'intensity_{detector + 1}'],
                       label=f'{experiment}_ratio_{detector}/{detector + 1}')

    ax[0].legend()
    ax[1].legend()
    plt.show()
