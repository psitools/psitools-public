# when run as MPI, this does a grid of calculations.
# usage mpirun -np 5 python3 -u testPISMode.py
#
import numpy as np
import time
from psitools import psi_mode_mpi
import pytest

# This really should have the min_size argument, but that seems to
# cause the test to not be collected.
@pytest.mark.mpi  # (min_size=2)
def test_psi_mode_mpi_0():
    stokes_range = [1e-8, 1e-0]
    dust_to_gas_ratio = 3.0

    real_range = [-2.0, 2.0]
    imag_range = [1e-8, 1.0]

    wall_start = time.time()
    wall_limit_total = 70*60*60

    # Set up the list of runs
    Kxaxis = np.logspace(-1, 3, 2)
    Kzaxis = np.logspace(-1, 3, 2)
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
                        'clean_tol': 1e-4},
                        'calculate': {
                            'wave_number_x': K[0],
                            'wave_number_z': K[1]},
                        'random_seed': 2})

    # get the MPI execution object
    ms = psi_mode_mpi.MpiScheduler(wall_start, wall_limit_total)

    # run the calculations
    finishedruns = ms.run(arglist)
# TODO:  This should check finishedruns correctness


@pytest.mark.mpi  # (min_size=2)
def test_psi_mode_mpi_1():

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
# TODO:  This should check finishedruns correctness
