import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

from data import NICKEL_REGIMES, load_experiment_data, get_nickel_references


def fit_experiments(runs, plot=-1):

    data = {f'{run.output_name or run.name}': load_experiment_data(run) for run in runs}
    references = get_nickel_references()

    coefficients = {}
    for i, (name, df) in enumerate(data.items()):

        detectors = runs[i].detectors

        if plot == i:
            fig, ax = plt.subplots(1, len(detectors), squeeze=False)
        else:
            ax = None

        for j, detector in enumerate(detectors):
            coefficients[(name, detector)] = fit_nickel_spectra(df, references, detector,
                                                                ax=None if ax is None else ax[0][j])

    if plot != -1:
        plt.show()

    return coefficients


def fit_nickel_spectra(df, references, detector, ax=None):

    mask_energy = (df['energy'] > NICKEL_REGIMES['fit_from']) & (df['energy'] < NICKEL_REGIMES['fit_to'])
    df = df.loc[mask_energy].reset_index(drop=True)

    # Compute reference spectra
    spectra = {name: spectrum(df['energy']) for name, spectrum in references.items()}
    names = list(spectra.keys())

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
    coefs, errors = non_negative_linear_model(X, residual)

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
            ax.plot(df['energy'], contribution, label=f'{name}: {coefficients[name]:.3g}', color=colors[i % 4])
            ax.plot(df['energy'], contribution + std, linestyle='dashed', color=colors[i % 4])
            ax.plot(df['energy'], contribution - std, linestyle='dashed', color=colors[i % 4])

        suffix = ''
        if len(names) == 2:
            # Assumes Ni2+ is first
            suffix = f'; ni2+/ni ratio: {coefficients[names[0]] / coefficients[names[1]]:.3g}'

        ax.plot(df['energy'], fit, label=f'fit{suffix}')
        ax.legend()

    return coefficients


def non_negative_linear_model(x, y):

    beta, rse = nnls(x, y)

    # Handle zeros for beta = 0.0 in error computation
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
