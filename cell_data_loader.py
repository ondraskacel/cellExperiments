from daxs.measurements import Hdf5Source, Xas

import matplotlib.pyplot as plt
import os
import pandas as pd

ENDPOINTS = 1

MAPPINGS = {
    "x": ".1/measurement/hdh_energy",
    "monitor": ".1/measurement/I02",
}

ROOT_DIR = '/data/visitor/ch6680/id26/20230510/RAW_DATA'


if __name__ == '__main__':
    
    experiments = {
        ('Ref-CCM-Naf212-Ni-Au-F_a', ''): list(range(5, 10)),
        ('Ref-CCM-Naf212-Ni-Au-G_a', ''): list(range(3, 8)),
        ('Ref-CCM-Naf212-Ni-Au-H_a', ''): list(range(3, 8)),
        ('Ref-CCM-Naf212-Ni-Au-I_a', '_hs'): list(range(8, 13)),
        ('Ref-CCM-Naf212-Ni-Au-I_a', '_no_hs'): list(range(13, 18)),
        ('Ref-CCM-Naf212-Ni-Au-I_a', '_new_y_1'): list(range(18, 23)),
        ('Ref-CCM-Naf212-Ni-Au-I_a', '_new_y_2'): list(range(23, 28)),
    }
    
    for (experiment, suffix), scans in experiments.items():
        path = f'{ROOT_DIR}/{experiment}/{experiment}_0001/{experiment}_0001.h5'
        
        for detector in range(1, 6):
            mapping = MAPPINGS.copy()
            mapping['signal'] = f'.1/measurement/p100k_eh2_roi_CA_{detector}'
            
            source = Hdf5Source(filename=path, included_scans=scans, data_mappings=mapping)
            measurement = Xas(source)
            
            # No normalization, we'll deal with the constants in another manner
            measurement.aggregate() 
            
            fig, ax = plt.subplots()
        
            for scan in measurement.scans:
                x = scan._x[:-ENDPOINTS]
                signal = scan._signal[0, :-ENDPOINTS]
                monitor = scan._monitor[:-ENDPOINTS]
                ax.plot(x, signal / monitor)
    
            ax.plot(measurement.x[:-ENDPOINTS], measurement.signal[:-ENDPOINTS], color='orange')
            
            plt.xlabel('Energy (keV)')
            plt.ylabel('Intensity (arb. units)')
            plt.title(f'{experiment}{suffix} detector {detector} scans={len(measurement.scans)}')
            plt.show()
            
            # Save the data
            df = pd.DataFrame({'energy': measurement.x[:-ENDPOINTS],
                               'intensity': measurement.signal[:-ENDPOINTS]})

            output_path = f'experiment_data/{experiment}_{detector}{suffix}.pickle'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_pickle(output_path)
