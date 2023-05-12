import matplotlib.pyplot as plt
import numpy as np

from experiment_setup import CELLS_SECOND_BATCH, PELLETS_SECOND_BATCH
from data import load_experiment_data


if __name__ == '__main__':

    cell = load_experiment_data(CELLS_SECOND_BATCH[0])
    ex_situ = load_experiment_data(PELLETS_SECOND_BATCH[9])

    fig, ax = plt.subplots(2)

    ax[0].plot(cell.filter(regex='intensity'))
    ax[1].plot(ex_situ.filter(regex='intensity'))

    plt.show()
