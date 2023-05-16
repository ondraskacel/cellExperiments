from experiment_setup import Experiment

import h5py
import hdf5plugin

import numpy as np
import os


if __name__ == '__main__':
    
    for scan in [2, 48]:
    
        experiment = Experiment(name='Ni-foil-thin', input_suffix='_0001', scans=(scan, ))
        path = experiment.path()

        signal = 'p100k_eh2'
        
        with h5py.File(path, 'r') as f:
            data = f[f'{scan}.1']['measurement'][signal]
            data_np = np.empty(data.shape)
            data.read_direct(data_np)
        
        output_path = f'geometry_data/scan_{scan}.npy'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, data_np)
        