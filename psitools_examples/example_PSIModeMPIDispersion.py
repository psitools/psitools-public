# When run as MPI, this does a grid of calculations.
# Usage mpirun -np 5 python3 -u example_PSIModeMPIDispersion.py
#
import numpy as np
import time
import h5py
import psitools.psi_mode_mpi

batchname = 'testPSIModeMPIDispersion'
datafile_hdf5 = batchname+'.hdf5'

stokes_range = [1e-8, 1e-0]
dust_to_gas_ratio = 3.0

real_range = [-2.0, 2.0]
imag_range = [-2.0, 2.0]

wave_number_x = 1e2
wave_number_z = 1e1


wall_start = time.time()
wall_limit_total = 70*60*60


# Set up the list of runs
Raxis = np.linspace(real_range[0], real_range[1], 32)
Iaxis = np.linspace(imag_range[0], imag_range[1], 32)
Rgrid, Igrid = np.meshgrid(Raxis, Iaxis)
Zs = list(zip(Rgrid.flatten(), Igrid.flatten()))
# arglist is a dict of dicts, with dicts for each method call
#  (to __init__ and caculate of PSIMode
#  random_seed is seperate
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
ms = psitools.psi_mode_mpi.DispersionRelationMpiScheduler(wall_start, wall_limit_total)

# run the calculations
finishedruns = ms.run(arglist)

# master process write an HDF5 file
if finishedruns is not None:
    # this is the master process
    Rgridout = np.zeros_like(Rgrid)
    Igridout = np.zeros_like(Igrid)
    realpart = np.zeros_like(Rgrid)
    imagpart = np.zeros_like(Rgrid)

    flatindex = 0
    for iimag, imag in enumerate(Iaxis):
        for ireal, real in enumerate(Raxis):
            args = arglist[flatindex]
            results = finishedruns[flatindex]
            Rgridout[iimag, ireal] = args['dispersion']['w'].real
            Igridout[iimag, ireal] = args['dispersion']['w'].imag
            realpart[iimag, ireal] = results.real
            imagpart[iimag, ireal] = results.imag
            flatindex += 1

    with h5py.File(datafile_hdf5, 'w') as h5f:
        grp = h5f.create_group(batchname)
        grp.attrs['stokes_range'] = stokes_range
        grp.attrs['dust_to_gas_ratio'] = dust_to_gas_ratio
        grp.attrs['real_range'] = real_range
        grp.attrs['imag_range'] = imag_range
        grp.attrs['wave_number_x'] = wave_number_x
        grp.attrs['wave_number_z'] = wave_number_z
        dset = grp.create_dataset('Raxis', data=Raxis)
        dset = grp.create_dataset('Iaxis', data=Iaxis)
        dset = grp.create_dataset('Rgrid', data=Rgridout)
        dset = grp.create_dataset('Igrid', data=Igridout)
        dset = grp.create_dataset('dispersion_real', data=realpart)
        dset = grp.create_dataset('dispersion_imag', data=imagpart)
        h5f.close()
