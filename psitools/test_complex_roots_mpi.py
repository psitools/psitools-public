# when run as MPI, this does a grid of calculations.
# usage mpirun -np 5 python3 -u testPISMode.py
#
import numpy as np
import time
from psitools import complex_roots_mpi
from psitools import tanhsinh
from mpi4py import MPI
import pytest


@pytest.mark.mpi  # (min_size=2)
def test_complex_roots_mpi_0():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    stokes_range = [1e-8, 1e-0]
    dust_to_gas_ratio = 3.0

    domain_real = [-2.0, 2.0]
    domain_imag = [2e-7, 2.0]

    z_list = [domain_real[0] + 1j*domain_imag[0],
              domain_real[1] + 1j*domain_imag[0],
              domain_real[1] + 1j*domain_imag[1],
              domain_real[0] + 1j*domain_imag[1]]

    wall_start = time.time()
    wall_limit_total = 70*60*60

    # Set up the list of runs
    Kxaxis = np.logspace(-1, 3, 3)
    Kzaxis = np.logspace(-1, 3, 3)
    Kxgrid, Kzgrid = np.meshgrid(Kxaxis, Kzaxis)
    Ks = list(zip(Kxgrid.flatten(), Kzgrid.flatten()))
    arglist = []
    for K in Ks:
        arglist.append({'PSIDispersion.__init__': {
                            'stokes_range': stokes_range,
                            'dust_to_gas_ratio': dust_to_gas_ratio,
                            'size_distribution_power': 3.5,
                            'tanhsinh_integrator':
                                 tanhsinh.TanhSinhNoDeepCopy()},
                        'dispersion': {
                            'wave_number_x': K[0],
                            'wave_number_z': K[1],
                            'viscous_alpha': 0.0},
                        'z_list': z_list,
                        'count_roots': {'tol': 0.1}
                        })

    # get the MPI execution object
    ms = complex_roots_mpi.MpiScheduler(wall_start, wall_limit_total)

    # run the calculations
    finishedruns = ms.run(arglist)
    if rank == 0:
        print('finishedruns is')
        for run in finishedruns:
            print('{:d}: '.format(run))
            print(finishedruns[run])
#finishedruns is
# 2:
# 0
# 1:
# 0
# 0:
# 0
# 3:
# 1
# 5:
# 0
# 4:
# 0
# 8:
# 0
# 7:
# 0
# 6:
# 1
        ref = [0, 0, 0, 1, 0, 0, 1, 0, 0]
        for i, count in enumerate(ref):
            assert(count == finishedruns[i])
