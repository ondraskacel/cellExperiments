from daxs.measurements import Hdf5Source, Xas

import matplotlib.pyplot as plt
import pandas as pd

MAPPINGS = {
    "x": ".1/measurement/hdh_energy",
    "signal": ".1/measurement/p100k_eh2_roi_xes",
    "monitor": ".1/measurement/I02",
}

ROOT_DIR = '/data/visitor/ch6680/id26/20230510/RAW_DATA'

SCANS = list(range(3, 83))


if __name__ == '__main__':
    
    dry_run = False

    pellets = [
        '04LS-2_4mg',
        '04LS-3mg',
        '09LS',
        'GRSP2-2_1mg',
        'GRSP2-6_3mg',
        'Ni(Ac)2',
        'Ni(AcAc)2',
        'Ni(OH)2',
        'NiO',
        'NiSO4-6H2O',
    ]
    
    suffixes = ['_0002' if pellet == 'NiO' else '' for pellet in pellets]
    
    outlier_indices = [
        [],
        [],
        list(range(32, 36)),
        [],
        [],
        list(range(16, 20)),
        list(range(56, 60)),
        [],
        [],
        list(range(60, 64)),
    ]
    
    for pellet, suffix, outliers in zip(pellets, suffixes, outlier_indices):
        
        dir_name = f'{pellet}_xanes_damage{suffix}'
        path = f'{ROOT_DIR}/{pellet}/{dir_name}/{dir_name}.h5'
        
        included_scans = [scan for idx, scan in enumerate(SCANS) if idx not in outliers]
        
        source = Hdf5Source(filename=path, included_scans=included_scans, data_mappings=MAPPINGS)
        measurement = Xas(source)
        
        fig, ax = plt.subplots()
        
        if dry_run:
            
            for scan in measurement.scans:
                ax.plot(scan._x, scan._signal[0, :] / scan._monitor)
            
        else:
            
            measurement.aggregate()
            measurement.normalize()
            
            ax.plot(measurement.x, measurement.signal)

        plt.xlabel('Energy (keV)')
        plt.ylabel('Intensity (arb. units)')
        plt.title(f'{pellet} scans={len(measurement.scans)}')
        plt.show()
        
        # Save the data
        if not dry_run:
            
            df = pd.DataFrame({'energy': measurement.x,
                               'intensity': measurement.signal})
            
            output_path = f'pellet_data/{pellet}.pickle'
            df.to_pickle(output_path)
