import numpy as np
from scipy.optimize import nnls

from data import NICKEL_REGIMES


def get_density(beta: np.ndarray):

    x_grid = np.linspace(0.0, 1.0, 51)

    log_density = beta[0]
    log_density += beta[1] * (2 * x_grid - 1)
    log_density += beta[2] * (6 * np.square(x_grid) - 6 * x_grid + 1)

    return np.exp(log_density)


def fit_nickel_spectra(df, references, detector, fit_nio=False, ax=None):

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

    # Fit the rest - errors are computed independently of the background fitting error (correct in practice)
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
        for i, name in enumerate(names + ['background']):

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

    # Handle zeros for beta = 0.9 in error computation
    # We set the errors for them to 0 and compute the others as if the model had only the non_zero regressors
    mask_zero = np.isclose(beta, 0.0)
    xtx = x.T @ x

    # Make the correlation matrix block-wise (numerical trick)
    xtx[mask_zero, :] = 0.0
    xtx[:, mask_zero] = 0.0
    np.fill_diagonal(xtx, np.where(mask_zero, 1.0, np.diag(xtx)))

    # Error computation
    dof = len(x) - np.sum(~mask_zero)
    xtx_inv = np.linalg.inv(xtx)

    errors = np.square(rse) / dof * xtx_inv

    # Zero out errors where beta = 0.0
    np.fill_diagonal(errors, np.where(mask_zero, 0.0, np.diag(errors)))

    return beta, errors
