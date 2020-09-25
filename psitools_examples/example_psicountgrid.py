# An example of using the root counter
# Run as MPI, use "python -u" for unbufferd output
#
import numpy as np
import time
from psitools import complex_roots_mpi
from psitools import tanhsinh
from mpi4py import MPI
import h5py


def count_grid(dust_to_gas_ratio, stokes_range, batchname):
    datafile_hdf5 = batchname + '.hdf5'
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    domain_real = [-2.0, 2.0]
    domain_imag = [2e-7, 2.0]

    z_list = [domain_real[0] + 1j*domain_imag[0],
              domain_real[1] + 1j*domain_imag[0],
              domain_real[1] + 1j*domain_imag[1],
              domain_real[0] + 1j*domain_imag[1]]

    wall_start = time.time()
    wall_limit_total = 70*60*60

    nk = 4
    Kxaxis = np.logspace(-1, 3, nk)
    Kzaxis = np.logspace(-1, 3, nk)

    integrator = tanhsinh.TanhSinhNoDeepCopy()

    Kxgrid, Kzgrid = np.meshgrid(Kxaxis, Kzaxis)
    Ks = list(zip(Kxgrid.flatten(), Kzgrid.flatten()))
    arglist = []
    for K in Ks:
        arglist.append({
            'PSIDispersion.__init__': {
                'stokes_range': stokes_range,
                'dust_to_gas_ratio': dust_to_gas_ratio,
                'size_distribution_power': 3.5,
                'tanhsinh_integrator': integrator},
            'dispersion': {
                'wave_number_x': K[0],
                'wave_number_z': K[1],
                'viscous_alpha': 0.0},
            'z_list': z_list,
            'count_roots': {'tol': 0.1}
            })

    ms = complex_roots_mpi.MpiScheduler(wall_start, wall_limit_total)
    finishedruns = ms.run(arglist)
    if rank == 0:
        rootcount = np.zeros(Kxgrid.shape, dtype=np.int)
        Kxgridout = np.zeros_like(Kxgrid)
        Kzgridout = np.zeros_like(Kzgrid)
        flatindex = 0
        for iz, kz in enumerate(Kzaxis):
            for ix, kx in enumerate(Kxaxis):
                args = arglist[flatindex]
                rootcount[iz, ix] = finishedruns[flatindex]
                Kxgridout[iz, ix] = args['dispersion']['wave_number_x']
                Kzgridout[iz, ix] = args['dispersion']['wave_number_z']
                flatindex += 1

        with h5py.File(datafile_hdf5, 'w') as h5f:
            grp = h5f.create_group(batchname)
            grp.attrs['stokes_range'] = \
                args['PSIDispersion.__init__']['stokes_range']
            grp.attrs['dust_to_gas_ratio'] = \
                args['PSIDispersion.__init__']['dust_to_gas_ratio']
            grp.attrs['domain_real'] = domain_real
            grp.attrs['domain_imag'] = domain_imag
            dset = grp.create_dataset('Kxaxis', data=Kxaxis)
            dset = grp.create_dataset('Kzaxis', data=Kzaxis)
            dset = grp.create_dataset('Kxgrid', data=Kxgridout)
            dset = grp.create_dataset('Kzgrid', data=Kzgridout)
            dset = grp.create_dataset('rootcount', data=rootcount)
            h5f.close()

# do something
count_grid(1.0, (1e-4, 1e-1), 'test_psicount')
