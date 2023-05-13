from daxs.measurements import Hdf5Source, Xas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


_POINT_RANGE = slice(1, -1)


def analyze_experiment(experiment, sum_over_detectors=False, **kwargs):
    
    path = experiment.path()
    scans = experiment.included_scans()
    
    signals_by_detector = []
    for detector in experiment.detectors:
        
        mapping = experiment.mapping(detector)
        source = Hdf5Source(filename=path, included_scans=scans, data_mappings=mapping)
        
        full_name = f'{experiment.name}{experiment.output_suffix}'
        plot_title = f'{full_name} detector {detector} scans={len(scans)}'
        
        output_path = experiment.output_path(detector)
        
        x, signal, monitor = analyze_source(source, output_path, plot_title, **kwargs)
        signals_by_detector.append(signal)
        
    if sum_over_detectors:
        total_signal = np.array(signals_by_detector).sum(axis=0)
        
        # All monitors are the same -> we use the last one
        df = pd.DataFrame({'energy': x[_POINT_RANGE],
                           'intensity': (total_signal / monitor)[_POINT_RANGE]})
        
        plt.plot(df['energy'], df['intensity'])
        plt.show()
        
        output_path = experiment.output_path('total')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_pickle(output_path)


def analyze_source(source, output_path, plot_title,
                   normalize=False, save=True, plot_scans=True, plot_beam_damage=False):
    
    measurement = Xas(source)
    measurement.aggregate() 
    
    if plot_scans:
        _plot_scans(measurement, plot_title)
        
    if plot_beam_damage:
        _plot_beam_damage(measurement, plot_title)
        
    if normalize:
        measurement.normalize()
    
    if save:
        df = pd.DataFrame({'energy': measurement.x[_POINT_RANGE],
                           'intensity': measurement.signal[_POINT_RANGE]})
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_pickle(output_path)
    
    # Return the denormalized signal
    return measurement.x, measurement.signal * measurement.monitor, measurement.monitor
    
    
def _plot_scans(measurement, title):
    
    fig, ax = _setup_plot(title)

    for scan in measurement.scans:
        
        x = scan._x[_POINT_RANGE]
        signal = scan._signal[0, _POINT_RANGE]
        monitor = scan._monitor[_POINT_RANGE]
        ax.plot(x, signal / monitor)

    ax.plot(measurement.x[_POINT_RANGE], measurement.signal[_POINT_RANGE], color='orange')
    plt.show()
    

def _plot_beam_damage(measurement, title):
    
    fig, ax = _setup_plot(f'{title} beam damage')
    
    x = np.array([scan._x[_POINT_RANGE] for scan in measurement.scans])
    signals = np.array([scan._signal[0, _POINT_RANGE] for scan in measurement.scans])
    monitors = np.array([scan._monitor[_POINT_RANGE] for scan in measurement.scans])
    
    for i in range(4):
        y = signals[i::4].sum(axis=0) / monitors[i::4].sum(axis=0)
        ax.plot(x[0], y)
        
    plt.show()
        

def _setup_plot(title):
    
    fig, ax = plt.subplots()
    plt.xlabel('Energy (keV)')
    plt.ylabel('Intensity (arb. units)')
    plt.title(title)
    
    return fig, ax
                        

if __name__ == '__main__':
    
    from experiment_setup import PELLETS_THIRD_BATCH as pellets
        
    for pellet in pellets:
        analyze_experiment(pellet, sum_over_detectors=True)

