from dataclasses import dataclass
from functools import partial
from typing import Tuple

_MAPPINGS = {
    'x': '.1/measurement/hdh_energy',
    'monitor': '.1/measurement/I02',
}

_ROOT_DIR = '/data/visitor/ch6680/id26/20230510/RAW_DATA'

@dataclass
class Experiment:
    
    name: str
    scans: Tuple[int]
    detectors: Tuple[str] = tuple(f'CA_{i}' for i in [1, 2, 3, 4, 5])
    
    input_suffix: str = ''
    output_name: str = ''
    
    # (2, 5) <-> scans[2] and scans[5] are outliers
    outlier_indices: Tuple[int] = tuple()
        
    def path(self):
        
        dir_name = f'{self.name}{self.input_suffix}'
        path = f'{_ROOT_DIR}/{self.name}/{dir_name}/{dir_name}.h5'
        
        return path
    
    def included_scans(self):
        
        return [scan for idx, scan in enumerate(self.scans) if idx not in self.outlier_indices]
    
    def signal(self, detector):
        
        return f'.1/measurement/p100k_eh2_roi_{detector}'
    
    def output_path(self, detector):
        
        return f'experiment_data/{self.name}/spectrum_{self.output_name}_{detector}.pickle'
    
    def mapping(self, detector):
        
        mapping = _MAPPINGS.copy()
        mapping['signal'] = self.signal(detector)
        
        return mapping
        

# Shortcuts to specifying pellet experiments
Pellet = partial(Experiment, scans = tuple(range(3, 83)), input_suffix = '_xanes_damage')
Pellet_1 = partial(Pellet, detectors=('xes', ))

PELLETS_FIRST_BATCH = [
    Pellet_1(name='04LS-2_4mg'),
    Pellet_1(name='04LS-3mg'),
    Pellet_1(name='09LS', outlier_indices=(32, 33, 34, 35)),
    Pellet_1(name='GRSP2-2_1mg'),
    Pellet_1(name='GRSP2-6_3mg'),
    Pellet_1(name='Ni(Ac)2', outlier_indices=(16, 17, 18, 19)),
    Pellet_1(name='Ni(AcAc)2', outlier_indices=(56, 57, 58, 59)),
    Pellet_1(name='Ni(OH)2'),
    Experiment(name='NiO', scans=tuple(range(3, 83)),
               detectors=('xes', ), input_suffix='_xanes_damage_0002'),
    Pellet_1(name='NiSO4-6H2O', outlier_indices=(60, 61, 62, 63)),
]

PELLETS_SECOND_BATCH = [
    Pellet(name='PtNi-dealloyed'),
    Pellet(name='PtNi', outlier_indices=(75, 76)),
    Pellet(name='K2'),
    Pellet(name='K1'),
    Pellet(name='V2'),
    Pellet(name='GRSP2CCM'),
    Pellet(name='LS04-layer'),
    Pellet(name='GDL'),
    Pellet(name='Ni-foil', outlier_indices=(72, 73, 74, 75)),
    Pellet(name='LS04-CCM'),
    Pellet(name='PtNi-dealloyed-CCM'),
    Pellet(name='V30-CCM'),
    Pellet(name='K30-CCM'),
    Pellet(name='NiSO4-on-Naf112'),
]

PELLETS_THIRD_BATCH = [
    Pellet(name='NiO_2'),
    Pellet(name='NiSO4-on-Naf112_2'),
    Pellet(name='Ni-foil_2'),
]

ALL_PELLETS = PELLETS_FIRST_BATCH + PELLETS_SECOND_BATCH + PELLETS_THIRD_BATCH


# Shortcuts to specifying cell experiments
def Cell_1(name, **kwargs):
    return Experiment(name='Ref-CCM-Naf212-Ni-Au-{name}_a'.format(name=name),
                      input_suffix='_0001', **kwargs)


def Cell_2(name, **kwargs):
    return Experiment(name='op-CCM-{i}-LS04-{s}'.format(i=name[0], s=name[1]),
                      input_suffix='_0001', **kwargs)


CELLS_FIRST_BATCH = [
    Cell_1('F', scans=(5, 6, 7, 8, 9)), 
    Cell_1('G', scans=(3, 4, 5, 6, 7)),
    Cell_1('H', scans=(3, 4, 5, 6, 7)),
]
    
CELL_I = [
    Cell_1('I', scans=(8, 9, 10, 11, 12), output_name='hs'),
    Cell_1('I', scans=(13, 14, 15, 16, 17), output_name='no_hs'),
    Cell_1('I', scans=(18, 19, 20, 21, 22), output_name='new_y_1'),
    Cell_1('I', scans=(23, 24, 25, 26, 27), output_name='new_y_2'),
]


CELL_J = [
    Cell_2(name=('J', 'a'), scans=(19, 20, 21, 22, 23)),
]
    
CELL_K = [
    Cell_2(name=('K', 'b'), scans=tuple(range(7, 18)), output_name='ocp1'),
    Cell_2(name=('K', 'b'), scans=tuple(range(18, 29)), output_name='hold1'),
    Cell_2(name=('K', 'b'), scans=tuple(range(29, 40)), output_name='ocp2'),
    Cell_2(name=('K', 'b'), scans=tuple(range(40, 51)), output_name='hold2'),
]
    
CELL_L =[
    Cell_2(name=('L', 'c'), scans=tuple(range(7, 18)), output_name='ocp1'),
    Cell_2(name=('L', 'c'), scans=tuple(range(18, 30)), output_name='hold1', outlier_indices=(2, )),
    Cell_2(name=('L', 'c'), scans=tuple(range(30, 42)), output_name='ocp2', outlier_indices=(6, )),
    Cell_2(name=('L', 'c'), scans=tuple(range(42, 53)), output_name='hold2'),
    Cell_2(name=('L', 'c'), scans=tuple(range(53, 75)), output_name='ocp3'),
    Cell_2(name=('L', 'c'), scans=tuple(range(75, 97)), output_name='hold3'),
    Cell_2(name=('L', 'c'), scans=tuple(range(97, 119)), output_name='hold4'),
    Cell_2(name=('L', 'c'), scans=tuple(range(119, 141)), output_name='ocp4'),
]
    
CELL_N = [
    Cell_2(name=('N', 'e'), scans=tuple(range(4, 14)), output_name='hold1_middle'),
    Cell_2(name=('N', 'e'), scans=tuple(range(14, 24)), output_name='hold1'),
    Cell_2(name=('N', 'e'), scans=tuple(range(29, 39)), output_name='hold2'),
    Cell_2(name=('N', 'e'), scans=tuple(range(39, 49)), output_name='ocp1'),
    Cell_2(name=('N', 'e'), scans=tuple(range(49, 59)), output_name='hold3'),
    Cell_2(name=('N', 'e'), scans=tuple(range(59, 63)), output_name='hold3_efgh'),
    Cell_2(name=('N', 'e'), scans=(63, ), output_name='hold3_x'),
    Cell_2(name=('N', 'e'), scans=tuple(range(64, 68)), output_name='ocp2_efgh'),
    Cell_2(name=('N', 'e'), scans=(68, ), output_name='ocp2_x'),
    Cell_2(name=('N', 'e'), scans=tuple(range(73, 77)), output_name='ocp2_efgh_realigned'),
    Cell_2(name=('N', 'e'), scans=(77, ), output_name='ocp2_x_realigned'),
]

CELL_P = [
    Cell_2(name=('P', 'g'), scans=tuple(range(5, 17)), output_name='hold1'),
    Cell_2(name=('P', 'g'), scans=tuple(range(19, 31)), output_name='ocp1', outlier_indices=(11, )),
    Cell_2(name=('P', 'g'), scans=tuple(range(33, 45)), output_name='hold2'),
    Cell_2(name=('P', 'g'), scans=tuple(range(47, 59)), output_name='ocp2', outlier_indices=(11, )),
]

CELL_Q = [
    Cell_2(name=('Q', 'h'), scans=tuple(range(8, 20)), output_name='hold1'),
    Cell_2(name=('Q', 'h'), scans=tuple(range(22, 34)), output_name='ocp1'),
    Cell_2(name=('Q', 'h'), scans=tuple(range(36, 48)), output_name='hold2'),
    Cell_2(name=('Q', 'h'), scans=tuple(range(50, 62)), output_name='ocp2'),
    Cell_2(name=('Q', 'h'), scans=tuple(range(64, 76)), output_name='hold3'),
    Cell_2(name=('Q', 'h'), scans=tuple(range(78, 90)), output_name='ocp3'),
    Cell_2(name=('Q', 'h'), scans=tuple(range(92, 104)), output_name='hold4'),
    Cell_2(name=('Q', 'h'), scans=tuple(range(106, 118)), output_name='hold_after_ast'),
    Cell_2(name=('Q', 'h'), scans=tuple(range(120, 132)), output_name='ocp4'),
]

CELL_R = [
    Cell_2(name=('R', 'i'), scans=tuple(range(9, 21)), output_name='ocp1'),
    Cell_2(name=('R', 'i'), scans=tuple(range(23, 35)), output_name='hold1'),
    Cell_2(name=('R', 'i'), scans=tuple(range(37, 49)), output_name='quick_hold'),
    Cell_2(name=('R', 'i'), scans=tuple(range(37, 43)), output_name='quick_hold_1-6'),
    Cell_2(name=('R', 'i'), scans=tuple(range(43, 49)), output_name='quick_hold_7-12'),
    Cell_2(name=('R', 'i'), scans=tuple(range(37, 40)), output_name='quick_hold_1-3'),
    Cell_2(name=('R', 'i'), scans=tuple(range(40, 43)), output_name='quick_hold_4-6'),
    Cell_2(name=('R', 'i'), scans=tuple(range(43, 46)), output_name='quick_hold_7-9'),
    Cell_2(name=('R', 'i'), scans=tuple(range(46, 49)), output_name='quick_hold_10-12'),
    Cell_2(name=('R', 'i'), scans=tuple(range(51, 63)), output_name='hold2'),
    Cell_2(name=('R', 'i'), scans=tuple(range(65, 77)), output_name='ocp3'),
    Cell_2(name=('R', 'i'), scans=tuple(range(79, 91)), output_name='hold3'),
    Cell_2(name=('R', 'i'), scans=tuple(range(93, 105)), output_name='ocp4'),
    Cell_2(name=('R', 'i'), scans=tuple(range(107, 119)), output_name='hold4'),
    Cell_2(name=('R', 'i'), scans=tuple(range(121, 133)), output_name='ocp5'),
    Cell_2(name=('R', 'i'), scans=tuple(range(133, 145)), output_name='hold5_instant'),
    Cell_2(name=('R', 'i'), scans=tuple(range(145, 157)), output_name='ocp6_instant'),
    Cell_2(name=('R', 'i'), scans=tuple(range(157, 169)), output_name='hold6_instant'),
    Cell_2(name=('R', 'i'), scans=tuple(range(171, 183)), output_name='ocp7_instant'),
    Cell_2(name=('R', 'i'), scans=tuple(range(183, 195)), output_name='hold7_instant'),
    Cell_2(name=('R', 'i'), scans=tuple(range(195, 207)), output_name='ocp8_instant'),
    Cell_2(name=('R', 'i'), scans=tuple(range(207, 219)), output_name='hold8_instant'),
    Cell_2(name=('R', 'i'), scans=tuple(range(221, 233)), output_name='hold9'),
    Cell_2(name=('R', 'i'), scans=tuple(range(233, 245)), output_name='ocp9_1'),
    Cell_2(name=('R', 'i'), scans=tuple(range(245, 257)), output_name='ocp9_2'),
    Cell_2(name=('R', 'i'), scans=tuple(range(257, 269)), output_name='ocp9_3'),
    Cell_2(name=('R', 'i'), scans=tuple(range(271, 283)), output_name='ocp9_4'),
    Cell_2(name=('R', 'i'), scans=tuple(range(283, 295)), output_name='hold10_1'),
    Cell_2(name=('R', 'i'), scans=tuple(range(295, 307)), output_name='hold10_2'),
    Cell_2(name=('R', 'i'), scans=tuple(range(309, 321)), output_name='hold10_3'),
    Cell_2(name=('R', 'i'), scans=tuple(range(321, 333)), output_name='hold10_4'),
    Cell_2(name=('R', 'i'), scans=tuple(range(333, 345)), output_name='ocp10_1'),
    Cell_2(name=('R', 'i'), scans=tuple(range(345, 357)), output_name='ocp10_2'),
]

ALL_CELLS = CELLS_FIRST_BATCH + CELL_I + CELL_J + CELL_K + CELL_L
ALL_CELLS = ALL_CELLS + CELL_N + CELL_P + CELL_Q + CELL_R

NI_FOIL = [
    Experiment(name='Ni-foil-thin', scans=tuple(range(6, 45)),
               input_suffix='_0001', output_name='full_cell'),
    Experiment(name='Ni-foil-thin', scans=tuple(range(51, 90)),
               input_suffix='_0001', output_name='half_cell'),  
]

# hack to get reference spectrum
NI_TRANSMISSION = Experiment(name='sample',
                             input_suffix='_0001',
                             output_name='ni_transmission',
                             detectors=('xes', ),
                             scans=(32, ))
                             
NI_TRANSMISSION.mapping = lambda detector: {'x': '.1/measurement/hdh_energy',
                                            'signal': '.1/measurement/I00',
                                            'monitor': '.1/measurement/I01'}
