# when run as MPI, this does a grid of calculations.
# usage mpirun -np 5 python3 -u testPISMode.py
#
import numpy as np
import numpy.testing as npt
import time
from psitools import psi_mode_mpi
from psitools import tanhsinh
from mpi4py import MPI
import pytest

# Tolerance for tests
rtol = 1e-7

# This really should have the min_size argument, but that seems to
# cause the test to not be collected by pytest.
@pytest.mark.mpi  # (min_size=2)
def test_psi_mode_mpi_0():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    stokes_range = [1e-8, 1e-0]
    dust_to_gas_ratio = 3.0

    real_range = [-2.0, 2.0]
    imag_range = [1e-8, 1.0]

    wall_start = time.time()
    wall_limit_total = 70*60*60

    # Set up the list of runs
    Kxaxis = np.logspace(-1, 3, 3)
    Kzaxis = np.logspace(-1, 3, 3)
    Kxgrid, Kzgrid = np.meshgrid(Kxaxis, Kzaxis)
    Ks = list(zip(Kxgrid.flatten(), Kzgrid.flatten()))
    arglist = []
    for K in Ks:
        arglist.append({'__init__': {
                        'stokes_range': stokes_range,
                        'dust_to_gas_ratio': dust_to_gas_ratio,
                        'size_distribution_power': 3.5,  # MRN
                        'real_range': real_range,
                        'imag_range': imag_range,
                        'n_sample': 20,
                        'max_zoom_domains': 1,
                        'tol': 1.0e-13,
                        'clean_tol': 1e-4,
                        'tanhsinh_integrator': tanhsinh.TanhSinhNoDeepCopy()},
                        'calculate': {
                            'wave_number_x': K[0],
                            'wave_number_z': K[1]},
                        'random_seed': 2})

    # get the MPI execution object
    ms = psi_mode_mpi.MpiScheduler(wall_start, wall_limit_total)

    # run the calculations
    finishedruns = ms.run(arglist)
    if rank == 0:
        print('finishedruns is')
        for run in finishedruns:
            print('{:d}: '.format(run))
            for val in finishedruns[run]:
                print(' {:+15.14e} {:+15.14e}j'.format(val.real, val.imag))
# a good output
# finishedruns is
# 2:
# 3:
# -1.00487995396850e+00 +8.94641471307505e-04j
# 0:
# 1:
# 5:
# 6:
#  -1.00492456189630e+00 +9.03864208351477e-04j
# 8:
# 4:
# 7:
        npt.assert_allclose(finishedruns[3].real, [-1.00487995396850e+00],
                            rtol=rtol)
        npt.assert_allclose(finishedruns[3].imag, [+8.94641471307505e-04],
                            rtol=rtol)
        npt.assert_allclose(finishedruns[6].real, [-1.00492456189630e+00],
                            rtol=rtol)
        npt.assert_allclose(finishedruns[6].imag, [+9.03864208351477e-04],
                            rtol=rtol)


@pytest.mark.mpi  # (min_size=2)
def test_psi_mode_mpi_1():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    stokes_range = [1e-8, 1e-0]
    dust_to_gas_ratio = 3.0

    real_range = [-2.0, 2.0]
    imag_range = [-2.0, 2.0]

    wave_number_x = 1e2
    wave_number_z = 1e1

    wall_start = time.time()
    wall_limit_total = 70*60*60

    # Set up the list of runs
    Raxis = np.linspace(real_range[0], real_range[1], 2)
    Iaxis = np.linspace(imag_range[0], imag_range[1], 2)
    Rgrid, Igrid = np.meshgrid(Raxis, Iaxis)
    Zs = list(zip(Rgrid.flatten(), Igrid.flatten()))
    arglist = []
    for zval in Zs:
        arglist.append({'__init__': {
                            'stokes_range': stokes_range,
                            'dust_to_gas_ratio': dust_to_gas_ratio,
                            'real_range': real_range,
                            'imag_range': imag_range},
                        'dispersion': {
                            'w': zval[0] + 1j*zval[1],
                            'wave_number_x': wave_number_x,
                            'wave_number_z': wave_number_z,
                            'viscous_alpha': 0.0},
                        })

    # get the MPI execution object
    ms = psi_mode_mpi.DispersionRelationMpiScheduler(wall_start,
                                                     wall_limit_total)

    # run the calculations
    finishedruns = ms.run(arglist)
    if rank == 0:
        print('finishedruns is')
        for run in finishedruns:
            print('{:d}: '.format(run))
            print(' {:+15.14e} {:+15.14e}j'.format(
                  finishedruns[run].real, finishedruns[run].imag))

# 0:
# -1.72721516816478e+08 +5.32417172686456e+07j
# 1:
# -1.01101748168798e+08 +1.70115997051966e+07j
# 2:
# -1.41019417999197e+08 +1.07996337299929e+08j
# 3:
# -8.18799484396433e+07 +8.84422553318089e+07j
        npt.assert_allclose(finishedruns[0].real, [-1.72721516816478e+08],
                            rtol=rtol)
        npt.assert_allclose(finishedruns[0].imag, [+5.32417172686456e+07],
                            rtol=rtol)
        npt.assert_allclose(finishedruns[3].real, [-8.18799484396433e+07],
                            rtol=rtol)
        npt.assert_allclose(finishedruns[3].imag, [+8.84422553318089e+07],
                            rtol=rtol)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
