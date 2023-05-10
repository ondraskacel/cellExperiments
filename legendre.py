import matplotlib.pyplot as plt
from scipy.special import sh_legendre

import numpy as np


def get_legendre_basis(p):

    return [sh_legendre(i) for i in range(p+1)]


def compute_penalty(basis):

    d = len(basis)
    penalty = np.empty((d, d))

    for i, p in enumerate(basis):
        for j, q in enumerate(basis):
            integrand = np.polyder(p, m=2) * np.polyder(q, m=2)
            penalty[i, j] = np.polyint(integrand)(1)

    return penalty


def evaluate_basis(basis, x):

    return np.array([p(x) for p in basis]).T


if __name__ == '__main__':

    p = 25
    basis = get_legendre_basis(p)
    penalty_matrix = compute_penalty(basis)

    n = 101
    x_grid = np.linspace(0.0, 1.0, n)
    y = np.random.normal(size=n) + 0.4 * x_grid

    X = evaluate_basis(basis, x_grid)

    xtx = X.T @ X / n
    xty = X.T @ y / n

    fig, ax = plt.subplots()

    plt.plot(x_grid, y, label='data')

    for c in np.exp(np.linspace(np.log(0.000001), np.log(1.0), 10)):

        beta = np.linalg.solve(xtx + c * penalty_matrix, xty)
        y_hat = X @ beta

        plt.plot(x_grid, y_hat, label=f'{c: .5g}')

    plt.legend()
    plt.show()
