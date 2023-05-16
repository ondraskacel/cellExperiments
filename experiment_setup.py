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
    output_name: str = ''
    
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
            return f'experiment_data/{self.name}/spectrum_{self.output_name}.pickle'
        
        return f'experiment_data/{self.name}/spectrum_{self.output_name}_{detector}.pickle'
    
    def mapping(self, detector):
        
        mapping = _MAPPINGS.copy()
        mapping['signal'] = self.signal(detector)
        
        return mapping
        

_PELLET_SCANS = list(range(3, 83))
_PELLET_SUFFIX = '_xanes_damage'


def _pellet_1(**kwargs):
    return Experiment(scans=_PELLET_SCANS,
                      detectors=[None],
                      input_suffix=_PELLET_SUFFIX,
                      **kwargs)


PELLETS_FIRST_BATCH = [
    _pellet_1(name='04LS-2_4mg'),
    _pellet_1(name='04LS-3mg'),
    _pellet_1(name='09LS', outlier_indices=[32, 33, 34, 35]),
    _pellet_1(name='GRSP2-2_1mg'),
    _pellet_1(name='GRSP2-6_3mg'),
    _pellet_1(name='Ni(Ac)2', outlier_indices=[16, 17, 18, 19]),
    _pellet_1(name='Ni(AcAc)2', outlier_indices=[56, 57, 58, 59]),
    _pellet_1(name='Ni(OH)2'),
    Experiment(name='NiO', scans=_PELLET_SCANS, detectors=[None], input_suffix=f'{_PELLET_SUFFIX}_0002'),
    _pellet_1(name='NiSO4-6H2O', outlier_indices=[60, 61, 62, 63]),
]


def _pellet_2(**kwargs):
    return Experiment(scans=_PELLET_SCANS,
                      input_suffix=_PELLET_SUFFIX,
                      **kwargs)


PELLETS_SECOND_BATCH = [
    _pellet_2(name='PtNi-dealloyed'),
    _pellet_2(name='PtNi', outlier_indices=[75, 76]),
    _pellet_2(name='K2'),
    _pellet_2(name='K1'),
    _pellet_2(name='V2'),
    _pellet_2(name='GRSP2CCM'),
    _pellet_2(name='LS04-layer'),
    _pellet_2(name='GDL'),
    _pellet_2(name='Ni-foil', outlier_indices=[72, 73, 74, 75]),
    _pellet_2(name='LS04-CCM'),
    _pellet_2(name='PtNi-dealloyed-CCM'),
    _pellet_2(name='V30-CCM'),
    _pellet_2(name='K30-CCM'),
    _pellet_2(name='NiSO4-on-Naf112'),
]

PELLETS_THIRD_BATCH = [
    _pellet_2(name='NiO_2'),
    _pellet_2(name='NiSO4-on-Naf112_2'),
    _pellet_2(name='Ni-foil_2'),
]

ALL_PELLETS = PELLETS_FIRST_BATCH + PELLETS_SECOND_BATCH + PELLETS_THIRD_BATCH


_CELL_NAME = 'Ref-CCM-Naf212-Ni-Au-{name}_a'
_CELL_SUFFIX = '_0001'


def _cell_1(name, **kwargs):
    return Experiment(name=_CELL_NAME.format(name=name),
                      input_suffix=_CELL_SUFFIX,
                      **kwargs)


CELLS_FIRST_BATCH = [
    _cell_1('F', scans=[5, 6, 7, 8, 9]), 
    _cell_1('G', scans=[3, 4, 5, 6, 7]),
    _cell_1('H', scans=[3, 4, 5, 6, 7]),
]
    
CELL_I = [
    _cell_1('I', scans=[8, 9, 10, 11, 12], output_name='hs'),
    _cell_1('I', scans=[13, 14, 15, 16, 17], output_name='no_hs'),
    _cell_1('I', scans=[18, 19, 20, 21, 22], output_name='new_y_1'),
    _cell_1('I', scans=[23, 24, 25, 26, 27], output_name='new_y_2'),
]

_CELL_NAME_2 = 'op-CCM-{i}-LS04-{s}'


def _cell_2(name, **kwargs):
    return Experiment(name=_CELL_NAME_2.format(i=name[0], s=name[1]),
                      input_suffix=_CELL_SUFFIX,
                      **kwargs)


CELL_J = [
    _cell_2(name=('J', 'a'), scans=[19, 20, 21, 22, 23]),
]
    
CELL_K = [
    _cell_2(name=('K', 'b'), scans=list(range(7, 18)), output_name='ocp1'),
    _cell_2(name=('K', 'b'), scans=list(range(18, 29)), output_name='hold1'),
    _cell_2(name=('K', 'b'), scans=list(range(29, 40)), output_name='ocp2'),
    _cell_2(name=('K', 'b'), scans=list(range(40, 51)), output_name='hold2'),
]
    
CELL_L =[
    _cell_2(name=('L', 'c'), scans=list(range(7, 18)), output_name='ocp1'),
    _cell_2(name=('L', 'c'), scans=list(range(18, 30)), output_name='hold1', outlier_indices=[2]),
    _cell_2(name=('L', 'c'), scans=list(range(30, 42)), output_name='ocp2', outlier_indices=[6]),
    _cell_2(name=('L', 'c'), scans=list(range(42, 53)), output_name='hold2'),
    _cell_2(name=('L', 'c'), scans=list(range(53, 75)), output_name='ocp3'),
    _cell_2(name=('L', 'c'), scans=list(range(75, 97)), output_name='hold3'),
    _cell_2(name=('L', 'c'), scans=list(range(97, 119)), output_name='hold4'),
    _cell_2(name=('L', 'c'), scans=list(range(119, 141)), output_name='ocp4'),
]
    
CELL_N = [
    _cell_2(name=('N', 'e'), scans=list(range(4, 14)), output_name='hold1_middle'),
    _cell_2(name=('N', 'e'), scans=list(range(14, 24)), output_name='hold1'),
    _cell_2(name=('N', 'e'), scans=list(range(29, 39)), output_name='hold2'),
    _cell_2(name=('N', 'e'), scans=list(range(39, 49)), output_name='ocp1'),
    _cell_2(name=('N', 'e'), scans=list(range(49, 59)), output_name='hold3'),
    _cell_2(name=('N', 'e'), scans=list(range(59, 63)), output_name='hold3_efgh'),
    _cell_2(name=('N', 'e'), scans=[63], output_name='hold3_x'),
    _cell_2(name=('N', 'e'), scans=list(range(64, 68)), output_name='ocp2_efgh'),
    _cell_2(name=('N', 'e'), scans=[68], output_name='ocp2_x'),
    _cell_2(name=('N', 'e'), scans=list(range(73, 77)), output_name='ocp2_efgh_realigned'),
    _cell_2(name=('N', 'e'), scans=[77], output_name='ocp2_x_realigned'),
]

CELL_P = [
    _cell_2(name=('P', 'g'), scans=list(range(5, 17)), output_name='hold1'),
    _cell_2(name=('P', 'g'), scans=list(range(19, 31)), output_name='ocp1', outlier_indices=[11]),
    _cell_2(name=('P', 'g'), scans=list(range(33, 45)), output_name='hold2'),
    _cell_2(name=('P', 'g'), scans=list(range(47, 59)), output_name='ocp2', outlier_indices=[11]),
]

CELL_Q = [
    _cell_2(name=('Q', 'h'), scans=list(range(8, 20)), output_name='hold1'),
    _cell_2(name=('Q', 'h'), scans=list(range(22, 34)), output_name='ocp1'),
    _cell_2(name=('Q', 'h'), scans=list(range(36, 48)), output_name='hold2'),
    _cell_2(name=('Q', 'h'), scans=list(range(50, 62)), output_name='ocp2'),
    _cell_2(name=('Q', 'h'), scans=list(range(64, 76)), output_name='hold3'),
    _cell_2(name=('Q', 'h'), scans=list(range(78, 90)), output_name='ocp3'),
    _cell_2(name=('Q', 'h'), scans=list(range(92, 104)), output_name='hold4'),
    _cell_2(name=('Q', 'h'), scans=list(range(106, 118)), output_name='hold_after_ast'),
    _cell_2(name=('Q', 'h'), scans=list(range(120, 132)), output_name='ocp4'),
]

CELL_R = [
    _cell_2(name=('R', 'i'), scans=list(range(9, 21)), output_name='ocp1'),
    _cell_2(name=('R', 'i'), scans=list(range(23, 35)), output_name='hold1'),
    _cell_2(name=('R', 'i'), scans=list(range(37, 49)), output_name='quick_hold'),
    _cell_2(name=('R', 'i'), scans=list(range(37, 43)), output_name='quick_hold_1-6'),
    _cell_2(name=('R', 'i'), scans=list(range(43, 49)), output_name='quick_hold_7-12'),
    _cell_2(name=('R', 'i'), scans=list(range(37, 40)), output_name='quick_hold_1-3'),
    _cell_2(name=('R', 'i'), scans=list(range(40, 43)), output_name='quick_hold_4-6'),
    _cell_2(name=('R', 'i'), scans=list(range(43, 46)), output_name='quick_hold_7-9'),
    _cell_2(name=('R', 'i'), scans=list(range(46, 49)), output_name='quick_hold_10-12'),
    _cell_2(name=('R', 'i'), scans=list(range(51, 63)), output_name='hold2'),
    _cell_2(name=('R', 'i'), scans=list(range(65, 77)), output_name='ocp3'),
    _cell_2(name=('R', 'i'), scans=list(range(79, 91)), output_name='hold3'),
    _cell_2(name=('R', 'i'), scans=list(range(93, 105)), output_name='ocp4'),
    _cell_2(name=('R', 'i'), scans=list(range(107, 119)), output_name='hold4'),
    _cell_2(name=('R', 'i'), scans=list(range(121, 133)), output_name='ocp5'),
    _cell_2(name=('R', 'i'), scans=list(range(133, 145)), output_name='hold5_instant'),
    _cell_2(name=('R', 'i'), scans=list(range(145, 157)), output_name='ocp6_instant'),
    _cell_2(name=('R', 'i'), scans=list(range(157, 169)), output_name='hold6_instant'),
    _cell_2(name=('R', 'i'), scans=list(range(171, 182)), output_name='ocp7_instant'),
]

ALL_CELLS = CELLS_FIRST_BATCH + CELL_I + CELL_J + CELL_K + CELL_L
ALL_CELLS = ALL_CELLS + CELL_N + CELL_P + CELL_Q + CELL_R

NI_FOIL = [
    Experiment(name='Ni-foil-thin', scans=list(range(6, 45)),
               input_suffix=_CELL_SUFFIX, output_name='full_cell'),
    Experiment(name='Ni-foil-thin', scans=list(range(51, 90)),
               input_suffix=_CELL_SUFFIX, output_name='half_cell'),  
]

# hack to get reference spectrum
NI_TRANSMISSION = Experiment(name='sample',
                             input_suffix=_CELL_SUFFIX,
                             output_name='ni_transmission',
                             detectors=[None],
                             scans=[32],)
                             
NI_TRANSMISSION.mapping = lambda detector: {'x': '.1/measurement/hdh_energy',
                                            'signal': '.1/measurement/I00',
                                            'monitor': '.1/measurement/I01'}
