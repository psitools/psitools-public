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
    gr = PSIGridRefiner(batchname, baseargs=baseargs, nbase=(3, 3), reruns=1)
    gr.run_basegrid()
    gr.fill_in_grid()

    ref = {
           1: np.array([-1.00080611317862e+00 +7.19887678757324e-05j]),
           2: [],
           0: [],
           3: [],
           5: [],
           7: np.array([-1.07591266729685e+00 +5.78802746530775e-03j,
                        -1.07591266729705e+00 +5.78802746530644e-03j]),
           9: [],
           11: [],
           13: [],
           15: [],
           17: [],
           19: np.array([-1.00075684729444e+00 +6.75986971488110e-05j,
                         -1.00075684729444e+00 +6.75986971483880e-05j]),
           21: []}

    if rank == 0:
        print('Top level grid results')
        for key in gr.grids[-1]['results']:
            print(key, ': ', end='')
            for res in gr.grids[-1]['results'][key]:
                print('  {:+15.14e} + {:+15.14e}j'.format(
                      res.real, res.imag), end='')
            print(' ')
        for key in ref:
            assert(len(ref[key]) == len(gr.grids[-1]['results'][key]))
            if len(ref[key]) > 0:
                ref[key].sort(),
                gr.grids[-1]['results'][key].sort()
                npt.assert_allclose(ref[key],
                                    gr.grids[-1]['results'][key],
                                    rtol=rtol)
