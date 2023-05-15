import matplotlib.pyplot as plt
import numpy as np

from data import NICKEL_REGIMES, load_experiment_data
from experiment_setup import NI_FOIL
from intensity import output_intensity
from materials import Layer, Cell, KAPTON


def get_geometric_factor_ascan(plot=False):

    # Limits determined manually
    limits_x = [75, 160]
    limits_y = [[40, 90],
                [130, 180],
                [220, 270],
                [300, 350],
                [400, 450]]

    # Good scans determined manually
    # Selection is I02-independent
    scans = {
        'in_cell': (2, [21, 30]),
        'out_of_cell': (48, [6, 75]),
    }

    coefficients = {}
    for scan, (number, range) in scans.items():
        data = np.load(f'data/geometry_data/scan_{number}.npy')

        # Plot detector receptive field
        # fig, ax = plt.subplots(2)
        # ax[0].plot(data.sum(axis=(0, 2)))
        # ax[1].plot(data.sum(axis=(0, 1)))
        #
        # plt.show()

        crystal_data = [data[:, limits_x[0]:limits_x[1], limit_y[0]: limit_y[1]] for limit_y in limits_y]
        intensities = np.array([np.sum(crystal, axis=(1, 2)) for crystal in crystal_data]).T

        if plot:
            plt.plot(intensities)
            plt.show()

        intensities = intensities[range[0]:range[1] + 1]
        coefficients[scan] = intensities.sum(axis=0)
        coefficients[scan] /= coefficients[scan][0]

    return coefficients


def get_norm(energy, intensity):

    # Subtract background
    mask_background = energy < NICKEL_REGIMES['pre_edge']
    background = intensity.loc[mask_background].mean()

    intensity = intensity - background

    return np.trapz(intensity, energy)


def get_geometric_factors_energy_scans():

    cells = {'in_cell': load_experiment_data(NI_FOIL[0]),
             'out_of_cell': load_experiment_data(NI_FOIL[1])}

    detectors = [1, 2, 3, 4, 5]

    norms = {}
    for name, cell in cells.items():
        norms[name] = np.array([get_norm(cell['energy'], cell[f'intensity_{detector}']) for detector in detectors])
        norms[name] = norms[name] / norms[name][0]

    return norms


def get_theoretical_angle_distribution():

    ni_foil = Cell(layers=[KAPTON, Layer(depth=5.0, densities={'Ni': 8.908})])

    density = np.zeros(7000)
    density[6500:] = 1.0

    theta_in = np.pi / 180 * 31
    theta_out = np.pi / 180 * np.array([41, 34, 27, 20, 13])  # Detectors C1-5

    energies = np.array([8400])

    results = output_intensity(1.0, theta_in, theta_out, ni_foil, energies, 7480, density)
    results = results[0]

    results = results / results[0]
    return results


def get_geometric_factors():

    factors = get_geometric_factors_energy_scans()
    factors['theoretical'] = get_theoretical_angle_distribution()

    return factors
