import numpy as np
from scipy.optimize import nnls

from data import NICKEL_REGIMES


def get_density(beta: np.ndarray):

    x_grid = np.linspace(0.0, 1.0, 51)

    log_density = beta[0]
    log_density += beta[1] * (2 * x_grid - 1)
    log_density += beta[2] * (6 * np.square(x_grid) - 6 * x_grid + 1)

    return np.exp(log_density)


def fit_nickel_spectra(df, references, detector, ax=None):

    spectra = {name: spectrum(df['energy']) for name, spectrum in references.items()}
    target = df[f'intensity_{detector}'].values

    coefficients = {}
    # errors = {}

    # Fit background
    mask_background = (df['energy'] < NICKEL_REGIMES['pre_edge'])
    coefficients['background'] = target[mask_background.values].mean()  # noqa

    residual = target - coefficients['background']

    # Fit the 'metallic' part of the spectrum
    mask_metallic = (df['energy'] < NICKEL_REGIMES['mixed']) & (df['energy'] > NICKEL_REGIMES['pre_edge'])

    residual_metallic = residual[mask_metallic.values]  # noqa
    metallic_reference = spectra['Ni-metallic'][mask_metallic.values]  # noqa

    # Explicit linear regression
    coefficients['Ni-metallic'] = np.sum(residual_metallic * metallic_reference) / np.sum(np.square(metallic_reference))

    residual = residual - coefficients['Ni-metallic'] * spectra['Ni-metallic']

    # Fit NiO and NiSO4
    X = np.array([spectra[name] for name in ['NiO', 'NiSO4']]).T
    (coefficients['NiO'], coefficients['NiSO4']), _ = nnls(X, residual)

    if ax is not None:
        ax.plot(df['energy'], target, label='data')

        fit = np.zeros_like(df['energy'])
        for name, spectrum in spectra.items():
            contribution = spectrum * coefficients[name]
            fit += contribution
            ax.plot(df['energy'], contribution, label=name)

        fit += coefficients['background']

        ax.plot(df['energy'], fit, label='fit')
        ax.legend()

    return coefficients

