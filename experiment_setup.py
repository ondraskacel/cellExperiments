from dataclasses import dataclass, field
from typing import Any, List, Tuple

_MAPPINGS = {
    'x': '.1/measurement/hdh_energy',
    'monitor': '.1/measurement/I02',
}

_ROOT_DIR = '/data/visitor/ch6680/id26/20230510/RAW_DATA'

@dataclass
class Experiment:
    
    name: str
    scans: List[int]
    detectors: List[Any] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    outlier_indices: Tuple[int] = field(default_factory=list)
    input_suffix: str = ''
    output_suffix: str = ''
    
    def __post_init__(self):
        
        self.scans = self.scans.copy()  # To allow sharing between experiments
        
    def path(self):
        
        dir_name = f'{self.name}{self.input_suffix}'
        path = f'{_ROOT_DIR}/{self.name}/{dir_name}/{dir_name}.h5'
        
        return path
    
    def included_scans(self):
        
        return [scan for idx, scan in enumerate(self.scans) if idx not in self.outlier_indices]
    
    def signal(self, detector):
        
        if detector is None:
            return '.1/measurement/p100k_eh2_roi_xes'
        
        return f'.1/measurement/p100k_eh2_roi_CA_{detector}'
    
    def output_path(self, detector):
        
        if detector is None:
            return f'experiment_data/{self.name}{self.output_suffix}.pickle'
        
        return f'experiment_data/{self.name}_{detector}{self.output_suffix}.pickle'
    
    def mapping(self, detector):
        
        mapping = _MAPPINGS.copy()
        mapping['signal'] = self.signal(detector)
        
        return mapping
        
            

_PELLET_SCANS = list(range(3, 83))
_PELLET_SUFFIX = '_xanes_damage'

_PELLET_1 = lambda **kwargs: Experiment(scans=_PELLET_SCANS,
                                        detectors=[None],
                                        input_suffix=_PELLET_SUFFIX,
                                        **kwargs)

PELLETS_FIRST_BATCH = [
    _PELLET_1(name='04LS-2_4mg'),
    _PELLET_1(name='04LS-3mg'),
    _PELLET_1(name='09LS', outlier_indices=[32, 33, 34, 35]),
    _PELLET_1(name='GRSP2-2_1mg'),
    _PELLET_1(name='GRSP2-6_3mg'),
    _PELLET_1(name='Ni(Ac)2', outlier_indices=[16, 17, 18, 19]),
    _PELLET_1(name='Ni(AcAc)2', outlier_indices=[56, 57, 58, 59]),
    _PELLET_1(name='Ni(OH)2'),
    Experiment(name='NiO', scans=_PELLET_SCANS, detectors=[None], input_suffix=f'{_PELLET_SUFFIX}_0002'),
    _PELLET_1(name='NiSO4-6H2O', outlier_indices=[60, 61, 62, 63]),
]

_CELL_NAME = 'Ref-CCM-Naf212-Ni-Au-{name}_a'
_CELL_SUFFIX = '_0001'

_CELL_1 = lambda name, **kwargs: Experiment(name=_CELL_NAME.format(name=name),
                                            input_suffix=_CELL_SUFFIX,
                                            **kwargs)

CELLS_FIRST_BATCH = [
    _CELL_1('F', scans=[5, 6, 7, 8, 9]), 
    _CELL_1('G', scans=[3, 4, 5, 6, 7]),
    _CELL_1('H', scans=[3, 4, 5, 6, 7]),
    _CELL_1('I', scans=[8, 9, 10, 11, 12], output_suffix='_hs'),
    _CELL_1('I', scans=[13, 14, 15, 16, 17], output_suffix='_no_hs'),
    _CELL_1('I', scans=[18, 19, 20, 21, 22], output_suffix='_new_y_1'),
    _CELL_1('I', scans=[23, 24, 25, 26, 27], output_suffix='_new_y_2'),
]


