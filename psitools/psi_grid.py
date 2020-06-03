#!/usr/bin/python

import numpy as np
import h5py

from . import psi_mode as psim
from . import complex_roots as cr

class PSIGrid():
    def __init__(self, pm):
        self.pm = pm
        self.guess_flag = True
        self.n_roots_found = 0

    def func(self, x, z, guess_roots=[]):
        np.random.seed(2)
        return self.pm.calculate(wave_number_x=x,
                                 wave_number_z=z,
                                 viscous_alpha=0,
                                 guess_roots=guess_roots)

    def calculate(self, wave_number_x, wave_number_z,
                  dynamic_plotter=None):
        # Shorthand
        self.Kx = wave_number_x
        self.Kz = wave_number_z
        self.dynamic_plotter = dynamic_plotter

        self.result = np.zeros((len(self.Kx), len(self.Kz)),
                               dtype=np.complex128)
        self.checked_flag = np.zeros((len(self.Kx), len(self.Kz)),
                                     dtype=bool)

        for i in range(0, len(self.Kx)):
            for j in range(0, len(self.Kz)):
                if self.checked_flag[i, j] == False:
                    self.find_root(i, j)

        return self.result

    def find_root(self, i, j, guess_roots=[]):
        roots = self.func(self.Kx[i], self.Kz[j], guess_roots=guess_roots)

        if len(guess_roots) > 0:
            self.checked_flag[i, j] = True

        if len(roots) > 0:
            self.result[i, j] = roots[np.argmax(roots.imag)]
            print(('Found growing mode at Kx = {}, Kz = {}:'
                   ' {}'.format(self.Kx[i], self.Kz[j], roots)))
            self.n_roots_found += 1

            if self.dynamic_plotter is not None:
                f = np.log10(np.imag(self.result))
                self.dynamic_plotter.plot(f)

            if self.guess_flag == True:
                if (i > 0 and
                    self.result[i - 1, j] == 0):
                    self.find_root(i - 1, j, guess_roots=[self.result[i, j]])
                if (j > 0 and
                    self.result[i, j - 1] == 0):
                    self.find_root(i, j - 1, guess_roots=[self.result[i, j]])
                if (i < len(self.Kx) - 1 and
                    self.result[i + 1, j] == 0):
                    self.find_root(i + 1, j, guess_roots=[self.result[i, j]])
                if (j < len(self.Kz) - 1 and
                    self.result[i, j + 1] == 0):
                    self.find_root(i, j + 1, guess_roots=[self.result[i, j]])

        else:
            print(('No growing mode found for Kx = {},'
                   ' Kz = {}').format(self.Kx[i], self.Kz[j]))

    def postprocess(self, wave_number_x, wave_number_z,
                    mode_frequencies,
                    max_iter=1,
                    dynamic_plotter=None,
                    min_neighbours=1):
        # Shorthand
        self.Kx = wave_number_x
        self.Kz = wave_number_z
        self.dynamic_plotter = dynamic_plotter
        self.guess_flag = True

        self.result = np.copy(mode_frequencies)
        self.checked_flag = np.zeros((len(self.Kx), len(self.Kz)),
                                     dtype=bool)

        if self.dynamic_plotter is not None:
            f = np.log10(np.imag(self.result))
            self.dynamic_plotter.plot(f)

        for n in range(0, max_iter):
            print('Starting postprocess...')

            self.n_roots_found = 0

            for i in range(0, len(self.Kx)):
                for j in range(0, len(self.Kz)):
                    if self.result[i, j] == 0:
                        guess_roots = []
                        if (i > 0 and
                            self.result[i - 1, j] != 0):
                            guess_roots.append(self.result[i - 1, j])
                        if (j > 0 and
                            self.result[i, j - 1] != 0):
                            guess_roots.append(self.result[i, j - 1])
                        if (i < len(self.Kx) - 1 and
                            self.result[i + 1, j] != 0):
                            guess_roots.append(self.result[i + 1, j])
                        if (j < len(self.Kz) - 1 and
                            self.result[i, j + 1] != 0):
                            guess_roots.append(self.result[i, j + 1])

                        if len(guess_roots) >= min_neighbours:
                            self.find_root(i, j, guess_roots=guess_roots)
            print('Number of roots added: ', self.n_roots_found)
            if self.n_roots_found == 0:
                break

        return self.result

    def dump_to_hdf(self, wave_number_x, wave_number_z,
                    mode_frequencies,
                    batchname='psi_grid'):
        with h5py.File(batchname + '.hdf5', 'w') as h5f:
            grp = h5f.create_group(batchname)
            #grp.attrs['stokes_range'] = stokes_range
            #grp.attrs['dust_to_gas_ratio'] = dust_to_gas_ratio
            #grp.attrs['real_range'] = real_range
            #grp.attrs['imag_range'] = imag_range

            dset = grp.create_dataset('Kx', data=wave_number_x)
            dset = grp.create_dataset('Kz', data=wave_number_z)
            #dset = grp.create_dataset('Kxgrid', data=Kxgridout)
            #dset = grp.create_dataset('Kzgrid', data=Kzgridout)
            dset = grp.create_dataset('root_real',
                                      data=np.real(mode_frequencies))
            dset = grp.create_dataset('root_imag',
                                      data=np.imag(mode_frequencies))
            #dset = grp.create_dataset('error', data=error)
            h5f.close()

    def read_from_hdf(self, batchname='psi_grid'):
        with h5py.File(batchname + '.hdf5', 'r') as h5f:
            self.Kx = h5f[batchname]['Kx'][()]
            self.Kz = h5f[batchname]['Kz'][()]
            self.result = h5f[batchname]['root_real'][()] +\
              1j*h5f[batchname]['root_imag'][()]
            return self.Kx, self.Kz, self.result

    def double_size(self):
        Kx = np.logspace(np.log10(self.Kx[0]),
                         np.log10(self.Kx[-1]),
                         2*len(self.Kx)-1)
        Kz = np.logspace(np.log10(self.Kz[0]),
                         np.log10(self.Kz[-1]),
                         2*len(self.Kz)-1)

        result = np.zeros((len(Kx), len(Kz)), dtype=np.complex128)
        result[0::2,0::2] = self.result

        return Kx, Kz, result
