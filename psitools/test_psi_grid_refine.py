import numpy as np
import numpy.testing as npt
from .power_bump import PowerBump
from .sizedensity import SizeDensity
from .tanhsinh import TanhSinhNoDeepCopy
from .psi_grid_refine import PSIGridRefiner
from mpi4py import MPI
import pytest

# Tolerance for tests
rtol = 1e-7


@pytest.mark.mpi
def test_psi_grid_refine_0():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    dust_to_gas_ratio = 10.0
    aP = 1e-1
    batchname = 'test_psi_grid_refine_batch'

    amin = 1.0e-8
    aL = 2.0/3.0*aP
    aR = 1.56*aP
    bumpfac = 2.0

    stokes_range = [amin, aR]

    pb = PowerBump(amin=amin, aP=aP, aL=aL, aR=aR, bumpfac=bumpfac)
    sd = SizeDensity(pb.sigma0, [amin, aR])
    pole = pb.get_discontinuity()
    sd.poles = [pole]

    real_range = [-2.0, 2.0]
    imag_range = [1e-7, 1.0]

    baseargs = {'__init__': {
                             'stokes_range': stokes_range,
                             'dust_to_gas_ratio': dust_to_gas_ratio,
                             'size_density': sd,
                             'tanhsinh_integrator': TanhSinhNoDeepCopy(),
                             'real_range': real_range,
                             'imag_range': imag_range,
                             'n_sample': 20,
                             'max_zoom_domains': 0,
                             'tol': 1.0e-13,
                             'clean_tol': 1e-4},
                 'calculate': {
                             'wave_number_x': None,
                             'wave_number_z': None,
                             'guess_roots': []},
                 'random_seed': 2}
    gr = PSIGridRefiner(batchname, baseargs=baseargs,
                        krange=(1, 1.1), nbase=(2, 2), reruns=0)
    gr.run_basegrid()
    gr.fill_in_grid()

    ref = {
        3: np.array([+2.23593733157486e-01+7.27645195058569e-02j, ]),
        2: np.array([+2.04261522692461e-01+1.47416079676104e-02j, ]),
        1: np.array([]),
        0: np.array([]),
        7: np.array([+2.51511471052411e-01+3.26775891059286e-02j, ]),
        4: np.array([]),
        5: np.array([]),
        6: np.array([+2.01409966220081e-01+2.64583254131715e-03j, ]),
        8: np.array([]),
        9: np.array([]),
        10: np.array([]),
        11: np.array([+2.47195619989170e-01+1.14903957786625e-02j, ]),
        12: np.array([]),
        13: np.array([]),
        14: np.array([]),
        15: np.array([]),
        16: np.array([]),
        17: np.array([]),
        18: np.array([]),
        19: np.array([+1.78930803971633e-01+2.88951770031046e-03j, ]),
        20: np.array([+2.23724845732854e-01+4.12981868972458e-02j, ]),
        21: np.array([+1.79178332025986e-01+6.71115333068255e-05j, ]),
        22: np.array([+2.02261529894614e-01+6.85128565981026e-03j, ]),
        23: np.array([+2.26862232605130e-01+2.17310189749878e-02j, ]),
        24: np.array([+2.42670813880620e-01+5.54008264063874e-02j, ]),
        25: np.array([]),
        26: np.array([+2.24074958667231e-01+1.06055048190402e-02j, ]),
        27: np.array([]),
        28: np.array([+2.01446217704532e-01+6.21130341991063e-04j, ]),
        29: np.array([+2.22480151123142e-01+6.03456362423705e-03j,
                      +2.22480151137097e-01+6.03456362464517e-03j, ]),
        30: np.array([+2.50031292938778e-01+1.92340302090360e-02j,
                      +2.50031292936582e-01+1.92340302081933e-02j, ]),
        31: np.array([+2.21721375643978e-01+3.13388988496292e-03j, ]),
        32: np.array([]),
        33: np.array([]),
        34: np.array([]),
        35: np.array([]),
        36: np.array([]),
        37: np.array([]),
        38: np.array([]),
        39: np.array([]),
        40: np.array([]),
        41: np.array([]),
        42: np.array([+2.21557582381651e-01+1.48741240945490e-03j,
                      +2.21557582371578e-01+1.48741242128824e-03j,
                      +2.21557582375905e-01+1.48741239857499e-03j,
                      +2.21557582377539e-01+1.48741238994857e-03j, ]),
        43: np.array([]),
        44: np.array([]),
        45: np.array([]),
        46: np.array([]),
        47: np.array([+2.45524301535846e-01+7.65008471117724e-03j, ]),
        48: np.array([]),
        49: np.array([+2.21664105126036e-01+2.51958249124266e-04j,
                      +2.21664105130649e-01+2.51958208599592e-04j, ])}

    if rank == 0:
        print('Top level grid results')
        for key in gr.grids[-1]['results']:
            print(key, ': np.array([', end='')
            for res in gr.grids[-1]['results'][key]:
                print('{:+15.14e}{:+15.14e}j, '.format(
                      res.real, res.imag), end='')
            print(']),')
        for key in ref:
            assert(len(ref[key]) == len(gr.grids[-1]['results'][key]))
            if len(ref[key]) > 0:
                ref[key].sort(),
                gr.grids[-1]['results'][key].sort()
                npt.assert_allclose(ref[key],
                                    gr.grids[-1]['results'][key],
                                    rtol=rtol)
