import matplotlib.pyplot as plt
import numpy as np


def fit_one_axis(data, axis, eps):

    ratio = np.cumsum(data.sum(axis=1-axis)) / data.sum()
    low = (ratio < eps).sum()
    high = data.shape[axis] - (ratio > 1.0 - eps).sum() + 1

    return low, high


def fit_reference_crystal(full_data, frame, limits_x, limits_y):

    data = full_data[frame, limits_x[0]:limits_x[1], limits_y[0]:limits_y[1]]

    # Assume axis aligned setup
    low_x, high_x = fit_one_axis(data, 0, eps=0.005)
    low_y, high_y = fit_one_axis(data[low_x:high_x, :], 1, eps=0.03)  # y is less well-defined

    return data[low_x:high_x, low_y:high_y]


def fit_side_crystals(full_data, frame, limits_x, limits_y):

    data = full_data[frame, limits_x[0]:limits_x[1], limits_y[0]:limits_y[1]]
    low_y, high_y = fit_one_axis(data, 1, eps=0.03)  # y is less well-defined

    return data[:, low_y:high_y]


if __name__ == '__main__':

    full_data = np.load('data/experiment_data/geometry_scan.npy')

    fig, ax = plt.subplots(2)
    ax[0].plot(full_data.sum(axis=(0, 2)))
    ax[1].plot(full_data.sum(axis=(0, 1)))

    plt.show()

    limits_x = [75, 160]
    limits_y = [[40, 90],
                [130, 180],
                [220, 270],
                [300, 350],
                [400, 450]]

    frame = 27

    rectangles = {
        'reference': fit_reference_crystal(full_data, frame, limits_x, limits_y[0]),
    }

    for detector in [2, 3, 4, 5]:
        rectangles[str(detector)] = fit_reference_crystal(full_data, frame, limits_x, limits_y[detector-1])

    for name, rect in rectangles.items():
        plt.plot(rect.sum(axis=1), label=name)

    plt.legend()
    plt.show()





