# when run as MPI, this does a grid of calculations.
# usage mpirun -np 5 python3 -u testPISMode.py
#
import numpy as np
import time
import h5py
#  import psi_mode_mpi
from psitools import psi_mode_mpi

batchname = 'testPSIModeMPI'
datafile_hdf5 = batchname+'.hdf5'

stokes_range = [1e-8, 1e-0]
dust_to_gas_ratio = 3.0

real_range = [-2.0, 2.0]
imag_range = [1e-8, 1.0]


wall_start = time.time()
wall_limit_total = 70*60*60


# Set up the list of runs
Kxaxis = np.logspace(-1, 3, 4)
Kzaxis = np.logspace(-1, 3, 4)
Kxgrid, Kzgrid = np.meshgrid(Kxaxis, Kzaxis)
Ks = list(zip(Kxgrid.flatten(), Kzgrid.flatten()))
# arglist is a dict of dicts, with dicts for each method call
# (to __init__ and caculate of PSIMode) random_seed is seperate
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

# get the MPI ececution object
ms = psi_mode_mpi.MpiScheduler(wall_start, wall_limit_total)

# run the calculations
finishedruns = ms.run(arglist)

# master process write an HDF5 file
if finishedruns is not None:
    # this is the master process
    Kxgridout = np.zeros_like(Kxgrid)
    Kzgridout = np.zeros_like(Kzgrid)
    realpart = np.zeros_like(Kzgrid)
    imagpart = np.zeros_like(Kzgrid)
    error = np.zeros_like(Kzgrid, dtype=np.int)

    flatindex = 0
    for iz, kz in enumerate(Kzaxis):
        for ix, kx in enumerate(Kxaxis):
            args = arglist[flatindex]
            results = finishedruns[flatindex]
            Kxgridout[iz, ix] = args['calculate']['wave_number_x']
            Kzgridout[iz, ix] = args['calculate']['wave_number_z']
            if len(results) > 0:
                realpart[iz, ix] = results[0].real
                imagpart[iz, ix] = results[0].imag
            error[iz, ix] = len(results)
            # 0 is no roots, 1 if one root,
            # >1 if we are eonly returning the first
            flatindex += 1

    with h5py.File(datafile_hdf5, 'w') as h5f:
        grp = h5f.create_group(batchname)
        grp.attrs['stokes_range'] = stokes_range
        grp.attrs['dust_to_gas_ratio'] = dust_to_gas_ratio
        grp.attrs['real_range'] = real_range
        grp.attrs['imag_range'] = imag_range
        dset = grp.create_dataset('Kxaxis', data=Kxaxis)
        dset = grp.create_dataset('Kzaxis', data=Kzaxis)
        dset = grp.create_dataset('Kxgrid', data=Kxgridout)
        dset = grp.create_dataset('Kzgrid', data=Kzgridout)
        dset = grp.create_dataset('root_real', data=realpart)
        dset = grp.create_dataset('root_imag', data=imagpart)
        dset = grp.create_dataset('error', data=error)
        h5f.close()
