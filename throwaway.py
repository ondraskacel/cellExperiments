"""

    # Setup Legendre polynomials
    p = 10
    basis = get_legendre_basis(p)
    penalty_matrix = compute_penalty(basis)

    x_grid = np.linspace(0.0, 1.0, 101)
    N = evaluate_basis(basis, x_grid)

    T = np.array([output_intensity(1.0, theta_in, theta_out, absorption_coef, np.arange(101).astype(int) == i, depth) for i in range(101)])
    X = T.T @ N

    xtx = X.T @ X
    xty = X.T @ actual_intensity

    fig, ax = plt.subplots()
    for c in np.exp(np.linspace(np.log(0.000001), np.log(1.0), 10)):

        beta = np.linalg.solve(xtx + c * penalty_matrix, xty)
        ax.plot(x_grid, N @ beta, label=f'{c: .5g}')
        y_hat = X @ beta

    plt.legend()
    plt.show()
    
"""

"""
def output_intensity(input_intensity: float,
                     theta_in: float,
                     theta_out: np.ndarray,
                     baseline_absorption: float,
                     nickel_density: np.ndarray,
                     depth: float,
                     ):
    """
    Assumes:
    - nickel does not contribute to the overall absorption
    - absorption is the same for the ingoing and outgoing direction
    - absorption is independent of depth
    - the outgoing angle is a single number
    TODO: assess (and remove?) the assumptions
    """

    linear_decay = baseline_absorption * (1.0 / np.sin(theta_in) + 1.0 / np.sin(theta_out))

    # The sine comes from the change of variables length -> depth
    multiplier = input_intensity / np.sin(theta_in)

    grid_size = len(nickel_density)
    grid_spacing = depth / grid_size
    depth_grid = np.linspace(0.0, depth, grid_size)

    # First axis <-> depth, second axis <-> theta_out
    integrand = nickel_density[:, None] * np.exp(-depth_grid[:, None] * linear_decay[None, :])
    return simple_integration(multiplier * integrand * grid_spacing)




def simple_integration(x: np.ndarray):
    """
    Performs a simple '(1 + 2 + 2 + ... + 2 + 1) / norm' integration over the first axis
    """

    assert len(x) > 2

    norm = len(x) - 1
    total = np.sum(x, axis=0) - 0.5 * (x[0] + x[-1])
    return total / norm * len(x)
"""