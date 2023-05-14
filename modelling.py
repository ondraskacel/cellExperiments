import numpy as np
from scipy.optimize import nnls


def get_density(beta: np.ndarray):

    x_grid = np.linspace(0.0, 1.0, 51)

    log_density = beta[0]
    log_density += beta[1] * (2 * x_grid - 1)
    log_density += beta[2] * (6 * np.square(x_grid) - 6 * x_grid + 1)

    return np.exp(log_density)


def decompose_into_references(df, references, detector, weights):

    spectra = {name: spectrum(df['energy']) for name, spectrum in references.items()}

    X = np.ones((len(df['energy']), len(references) + 1))
    for i, intensity in enumerate(spectra.values()):
        X[:, i] = intensity

    y = df[f'intensity_{detector}'].values
    w = weights / np.square(y)

    xtx = X.T @ (w[:, None] * X)
    beta_ = np.linalg.solve(xtx, X.T @ (w * y))

    beta, err = nnls(X * np.sqrt(w[:, None]), y * np.sqrt(w), maxiter=1000000)

    y_hat = X @ beta
    chi = np.sum(w * np.square(y - y_hat)) / (len(y) - X.shape[1])
    errors = chi * np.linalg.inv(xtx)
    sd = np.sqrt(np.diag(errors))

    # if detector == 3:
    #     print(...)
    #
    # plt.plot(df['energy'], y, label='data')
    # plt.plot(df['energy'], X @ beta, label='fit')
    # plt.plot(df['energy'], X @ beta__, label='fit')
    #
    # colors = ['red', 'green', 'blue']
    # for i, name in enumerate(spectra.keys()):
    #     component = X[:, i] * beta[i]
    #     error = X[:, i] * sd[i]
    #     plt.plot(df['energy'], component, label=f'{name} component', color=colors[i])
    #     plt.plot(df['energy'], component + error, color=colors[i], linestyle='dashed')
    #     plt.plot(df['energy'], component - error, color=colors[i], linestyle='dashed')
    #
    # plt.legend()
    # plt.show()

    return beta, sd
