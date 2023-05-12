from daxs.measurements import Hdf5Source, Xas

import matplotlib.pyplot as plt
import pandas as pd
import os


_ENDPOINTS = 1


def analyze_experiment(experiment, normalize=False):
    
    path = experiment.path()
    scans = experiment.included_scans()
    
    for detector in experiment.detectors:
        
        mapping = experiment.mapping(detector)
        source = Hdf5Source(filename=path, included_scans=scans, data_mappings=mapping)
        
        full_name = f'{experiment.name}{experiment.input_suffix}'
        plot_title = f'{full_name} detector {detector} scans={len(scans)}'
        
        output_path = experiment.output_path(detector)
        
        analyze_source(source, output_path, plot_title, normalize)


def analyze_source(source, output_path, plot_title, normalize):
    
    measurement = Xas(source)
    measurement.aggregate() 
    
    plot_measurement(measurement, plot_title)
        
    if normalize:
        measurement.normalize()
    
    # Save the data
    df = pd.DataFrame({'energy': measurement.x[:-_ENDPOINTS],
                       'intensity': measurement.signal[:-_ENDPOINTS]})
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_pickle(output_path)
    
    
def plot_measurement(measurement, title):
    
    fig, ax = plt.subplots()

    for scan in measurement.scans:
        
        x = scan._x[:-_ENDPOINTS]
        signal = scan._signal[0, :-_ENDPOINTS]
        monitor = scan._monitor[:-_ENDPOINTS]
        ax.plot(x, signal / monitor)

    ax.plot(measurement.x[:-_ENDPOINTS], measurement.signal[:-_ENDPOINTS], color='orange')
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('Intensity (arb. units)')
    plt.title(title)
    plt.show()
                        


if __name__ == '__main__':
    
    from experiment_setup import CELLS_FIRST_BATCH, PELLETS_FIRST_BATCH
    
    for cell in CELLS_FIRST_BATCH:
        analyze_experiment(cell)
        
    for pellet in PELLETS_FIRST_BATCH:
        analyze_experiment(pellet)

