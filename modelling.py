import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

from data import NICKEL_REGIMES


def get_density(beta: np.ndarray):

    x_grid = np.linspace(0.0, 1.0, 51)

    log_density = beta[0]
    log_density += beta[1] * (2 * x_grid - 1)
    log_density += beta[2] * (6 * np.square(x_grid) - 6 * x_grid + 1)

    return np.exp(log_density)


def fit_nickel_spectra(df, references, detector, plot=True):

    spectra = {name: spectrum(df['energy']) for name, spectrum in references.items()}
    target = df[f'intensity_{detector}'].values

    coefficients = {}
    error_variance = {}

    # Fit background
    mask_background = (df['energy'] < NICKEL_REGIMES['pre_edge'])
    background = target[mask_background.values]  # noqa
    coefficients['background'] = background.mean()
    error_variance['background'] = background.var()

    residual = target - coefficients['background']

    # Fit the 'metallic' part of the spectrum
    mask_metallic = (df['energy'] < NICKEL_REGIMES['mixed']) & (df['energy'] > NICKEL_REGIMES['metallic'])

    residual_metallic = residual[mask_metallic.values]  # noqa
    metallic_reference = spectra['Ni-metallic'][mask_metallic.values]  # noqa

    # Linear regression
    coef, errors = nonnegative_linear_model(metallic_reference[:, None], residual_metallic)
    coefficients['Ni-metallic'] = coef[0]
    error_variance['Ni-metallic'] = errors[0, 0]

    residual = residual - coefficients['Ni-metallic'] * spectra['Ni-metallic']

    # Fit NiO and NiSO4
    X = np.array([spectra[name] for name in ['NiO', 'NiSO4']]).T
    coefs, errors = nonnegative_linear_model(X, residual)

    for i, name in enumerate(['NiO', 'NiSO4']):
        coefficients[name] = coefs[i]
        error_variance[name] = errors[i, i]

    if plot:
        plt.plot(df['energy'], target, label='data')

        fit = np.zeros_like(df['energy'])
        for name, spectrum in spectra.items():
            contribution = spectrum * coefficients[name]
            fit += contribution
            plt.plot(df['energy'], contribution, label=name)

        fit += coefficients['background']

        plt.plot(df['energy'], fit, label='fit')
        plt.legend()
        plt.show()

    return coefficients


def fit_nickel_spectra_lasso(df, references, detector, fit_nio=False, ax=None):

    names = [name for name in references.keys()]
    if not fit_nio:
        names.remove('NiO')

    # Compute reference spectra
    spectra = {name: references[name](df['energy']) for name in names}
    target = df[f'intensity_{detector}'].values

    coefficients = {}
    error_variance = {}

    # Fit background
    mask_background = (df['energy'] < NICKEL_REGIMES['pre_edge'])
    background = target[mask_background.values]  # noqa
    coefficients['background'] = background.mean()
    error_variance['background'] = background.var()

    residual = target - coefficients['background']

    # Fit the rest
    X = np.array([spectrum for spectrum in spectra.values()]).T
    coefs, errors = nonnegative_linear_model(X, residual)

    for i, name in enumerate(names):
        coefficients[name] = coefs[i]
        error_variance[name] = errors[i, i]

    # Optional plotting
    if ax is not None:

        ax.plot(df['energy'], target, label='data')

        fit = np.zeros_like(df['energy'])
        spectra['background'] = np.ones_like(df['energy'])

        colors = ['red', 'green', 'blue', 'black']
        for i, name in enumerate(names):

            contribution = coefficients[name] * spectra[name]
            std = np.sqrt(error_variance[name]) * spectra[name]

            fit += contribution
            ax.plot(df['energy'], contribution, label=name, color=colors[i])
            ax.plot(df['energy'], contribution + std, linestyle='dashed', color=colors[i])
            ax.plot(df['energy'], contribution - std, linestyle='dashed', color=colors[i])

        ax.plot(df['energy'], fit, label='fit')
        ax.legend()

    return coefficients


def nonnegative_linear_model(x, y):

    beta, rse = nnls(x, y)

    # TODO: handle zeros in error computation

    # Error computation
    dof = len(x) - x.shape[1]
    xtx_inv = np.linalg.inv(x.T @ x)

    errors = np.square(rse) / dof * xtx_inv

    return beta, errors
