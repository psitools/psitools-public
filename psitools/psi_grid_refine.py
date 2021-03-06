#
# A class to parallelize the refinement-smoothing algorithm for making growth
# maps
#
#
# Copyright 2020 Colin McNally, Sijme-Jan Paardekooper, Francesco Lovascio
#    colin@colinmcnally.ca, s.j.paardekooper@qmul.ac.uk, f.lovascio@qmul.ac.uk
#
#    This file is part of psitools.
#
#    psitools is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    psitools is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with psitools.  If not, see <https://www.gnu.org/licenses/>.
#
# Who: Colin's doing, from Sijme-Jan's prototype
import numpy as np
import scipy.cluster.vq
import time
import h5py
import copy
import tracemalloc
from mpi4py import MPI
from psitools import psi_mode_mpi
from psitools.psi_mode import guess_domain_size


def spreadgrid(old, const_axis=0):
    """Grid stretch with log10 insertion
    const_axis is the axis where the grid is constant for
    meshgrid( indxing='ij') with 2D grids this is the opposite
    of the return list index
    A, B = meshgrid(a, b, indexing='ij')
    A has const_axis=1
    B has const_axis=0"""
    new = np.zeros([2*l-1 for l in old.shape])
    new[::2, ::2] = old
    if const_axis == 0:
        new[1:-1:2, ::2] = new[:-1:2, ::2]
        new[:, 1:-1:2] = 10**(0.5*(np.log10(new[:, :-2:2])
                                   + np.log10(new[:, 2::2])))
    elif const_axis == 1:
        new[::2, 1:-1:2] = new[::2, :-1:2]
        new[1:-1:2, :] = 10**(0.5*(np.log10(new[:-2:2, :])
                                   + np.log10(new[2::2, :])))
    else:
        raise ValueError('Bad const_axis')
    return new


def get_if(finishedruns, runkey):
    """Utility for getting entries form a dict of lists, catching the key -1
    as meaning the list returned should be empty"""
    if runkey >= 0:  # if not -1, which means there should be a run
        return list(finishedruns[runkey])
    else:
        return []


def prune_eps(values, epsilon=1e-5):
    """For a crude pruning of lists of root guesses."""
    if values is None:
        return values
    if len(values) <= 1:
        return values
    uniques = [values[0]]
    for v in values[1:]:
        addme = True
        for vu in uniques:
            if np.abs(v-vu) < epsilon:
                addme = False
                break
        if addme:
            uniques.append(v)

    return np.array(uniques)


def prune_kmeans(guess_roots):
    pruned_guesses = guess_roots
    if len(guess_roots) > 2:
        obs = np.vstack((guess_roots.real, guess_roots.imag)).transpose()
        guess1, mdist1 = scipy.cluster.vq.kmeans(obs, 1)
        guess2, mdist2 = scipy.cluster.vq.kmeans(obs, 2)
        # Decide if one or two clusters, based on a crude coomparison
        if mdist1 < guess_domain_size(guess1[0, 0] + 1j*guess1[0, 1]):
            pruned_guesses = guess1[:, 0] + 1j*guess1[:, 1]
        else:
            pruned_guesses = guess2[:, 0] + 1j*guess2[:, 1]
    return pruned_guesses


class PSIGridRefiner:
    def __init__(self, batchname, baseargs=None, nbase=(16, 16),
                 reruns=3, verbose=False, krange=(-1.0, 3.0)):
        if baseargs:
            self.baseargs = baseargs
        else:
            # Some defaults for testing
            stokes_range = (1-8, 1e-1)
            dust_to_gas_ratio = 10.0
            real_range = [-2.0, 2.0]
            imag_range = [1e-8, 1.0]
            self.baseargs = {'__init__': {
                            'stokes_range': stokes_range,
                            'dust_to_gas_ratio': dust_to_gas_ratio,
                            'size_distribution_power': 3.5,  # MRN
                            'real_range': real_range,
                            'imag_range': imag_range,
                            'n_sample': 30,
                            'max_zoom_domains': 1,
                            'tol': 1.0e-13,
                            'clean_tol': 1e-4},
                'calculate': {
                            'wave_number_x': None,
                            'wave_number_z': None,
                            'guess_roots': []},
                'random_seed': 2}

        self.nbase = nbase
        self.krange = krange
        self.reruns = reruns
        self.batchname = batchname
        self.datafile_hdf5 = batchname+'.hdf5'

        self.verbose = verbose

        self.wall_start = time.time()
        self.wall_limit_total = 70*60*60

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        if self.rank == 0:
            self.root = True
        else:
            self.root = False
        # get the MPI excecution object
        self.ms = psi_mode_mpi.MpiScheduler(self.wall_start,
                                            self.wall_limit_total,
                                            verbose=self.verbose)
        self.grids = []

    def run_basegrid(self):
        # Set up the list of runs
        Kxaxis = np.logspace(self.krange[0], self.krange[1],
                             self.nbase[1])
        dK = Kxaxis[1] - Kxaxis[0]
        Kxaxis = np.hstack(([Kxaxis[0]-dK], Kxaxis, [Kxaxis[1]+dK]))
        Kzaxis = np.logspace(self.krange[0], self.krange[1],
                             self.nbase[0])
        dK = Kzaxis[1] - Kzaxis[0]
        Kzaxis = np.hstack(([Kzaxis[0]-dK], Kzaxis, [Kzaxis[1]+dK]))
        Kzgrid, Kxgrid = np.meshgrid(Kzaxis, Kxaxis, indexing='ij')
        rungrid = np.zeros(Kxgrid.shape, dtype=np.int)
        nguess = np.zeros(Kxgrid.shape, dtype=np.int)
        rungridflat = rungrid.ravel()
        Ks = list(zip(Kxgrid.ravel(), Kzgrid.ravel()))
        # arglist is a dict of dicts, with dicts for each method call
        # (to __init__ and caculate of PSIMode) random_seed is seperate
        arglist = []
        for ik, K in enumerate(Ks):
            args = copy.deepcopy(self.baseargs)
            args['__init__']['max_zoom_domains'] = 1
            args['calculate']['wave_number_x'] = K[0]
            args['calculate']['wave_number_z'] = K[1]
            arglist.append(args)
            rungridflat[ik] = ik  # redundant info, but explicit

        # run the calculations
        finishedruns = self.ms.run(arglist)

        # Broadcast to all procs
        finishedruns = self.comm.bcast(finishedruns, root=0)

        # all procs do this bookeeping
        self.grids.append({'Kx': Kxgrid, 'Kz': Kzgrid,
                           'runs': rungrid, 'results': finishedruns,
                           'nguess': nguess})
        max_sweep = 32
        for i in range(0, max_sweep):
            if self.sweep_last_grid() == 0:
                break

    def fill_in_grid(self):
        old = self.grids[-1]
        Kxgrid = spreadgrid(old['Kx'], const_axis=0)
        Kzgrid = spreadgrid(old['Kz'], const_axis=1)

        rungrid = -1*np.ones(Kxgrid.shape, dtype=np.int)
        nguess = np.zeros(Kxgrid.shape, dtype=np.int)
        # Now we need to insert the old rungrid into the new one
        for iz, Kz in enumerate(Kzgrid[:, 0]):
            if iz % 2 == 0:
                for ix, Kx in enumerate(Kxgrid[0, :]):
                    if ix % 2 == 0:
                        rungrid[iz, ix] = old['runs'][iz//2, ix//2]
        finishedruns = copy.deepcopy(old['results'])

        # all procs do this bookeeping
        self.grids.append({'Kx': Kxgrid, 'Kz': Kzgrid,
                           'runs': rungrid, 'results': finishedruns,
                           'nguess': nguess})
        if self.root:
            print(' running grid sweeps')
        max_sweep = 32
        for i in range(0, max_sweep):
            if self.sweep_last_grid() == 0:
                break

    def sweep_last_grid(self):
        if self.rank == 0:
            tracemalloc.start()
        Kxgrid = self.grids[-1]['Kx']
        Kzgrid = self.grids[-1]['Kz']
        rungrid = self.grids[-1]['runs']
        nguess = self.grids[-1]['nguess']
        finishedruns = self.grids[-1]['results']
        offset = max(finishedruns)
        # Test validity of mapping
        for key in rungrid.ravel():
            get_if(finishedruns, key)

        # Now we construct a new run list
        # Loop over the new points
        newrungrid = np.zeros_like(rungrid)
        arglist = []
        iarg = 0
        for (iz, ix), Kz in np.ndenumerate(Kzgrid):
            if len(get_if(finishedruns, rungrid[iz, ix])) > 0:
                continue  # Never rerun a point with roots
            near_roots = []
            if ix > 0 and iz > 0:
                near_roots += get_if(finishedruns, rungrid[iz-1, ix-1])
            if ix > 0:
                near_roots += get_if(finishedruns, rungrid[iz, ix-1])
            if ix > 0 and iz < Kzgrid.shape[0]-1:
                near_roots += get_if(finishedruns, rungrid[iz+1, ix-1])
            if iz > 0:
                near_roots += get_if(finishedruns, rungrid[iz-1, ix])
            if iz < Kzgrid.shape[0]-1:
                near_roots += get_if(finishedruns, rungrid[iz+1, ix])
            if ix < Kxgrid.shape[1]-1 and iz > 0:
                near_roots += get_if(finishedruns, rungrid[iz-1, ix+1])
            if ix < Kxgrid.shape[1]-1:
                near_roots += get_if(finishedruns, rungrid[iz, ix+1])
            if ix < Kxgrid.shape[1]-1 and iz < Kzgrid.shape[0]-1:
                near_roots += get_if(finishedruns, rungrid[iz+1, ix+1])
            #prune based on the local growth rate
            if len(near_roots) > 0:
                near_roots = prune_eps(near_roots,
                                 epsilon=np.array(near_roots).imag.mean()*1e-4)
            foundguesses = len(near_roots)
            # rerun only is more guesses available
            if foundguesses > nguess[iz, ix]:
                nguess[iz, ix] = foundguesses
                args = copy.deepcopy(self.baseargs)
                args['__init__']['max_zoom_domains'] = 0
                args['calculate']['wave_number_x'] = Kxgrid[iz, ix]
                args['calculate']['wave_number_z'] = Kzgrid[iz, ix]
                # It is a little non-optimal to do k-means on every
                # proc for every list...
                args['calculate']['guess_roots'] = \
                    near_roots
                    # K-means pruning seems to be too agressive
                    # prune_kmeans(prune_eps(near_roots))
                #if self.root and self.verbose \
                #   and len(prune_eps(near_roots)) > 2:
                #    print('Original guess_roots ', prune_eps(near_roots))
                #    print('K-means pruned ',
                #          prune_kmeans(prune_eps(near_roots)))
                arglist.append(args)
                newrungrid[iz, ix] = offset + iarg - rungrid[iz, ix]
                firstiarg = iarg
                iarg += 1
                for ia in range(0, self.reruns):
                    args = copy.deepcopy(args)
                    args['random_seed'] += 1
                    args['is_rerun'] = firstiarg  # can hide a flag here
                    arglist.append(args)
                    # Don't set rungrid,
                    # just append the results to the original run roots.
                    iarg += 1

        if self.rank == 0:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            print('--------------------------')
            print("[ Tracemalloc Top 10 ]")
            for stat in top_stats[:10]:
                print(stat)
            print('--------------------------')
            print('Will run ', iarg, ' new points')

        # Run the calculations
        newroots = 0
        if iarg > 0:
            newfinishedruns = self.ms.run(arglist)
            if self.root:
                for irun in sorted(newfinishedruns):
                    # Have to be careful as the dict can enumerate in any order
                    if 'is_rerun' in arglist[irun]:
                        if len(finishedruns[offset+arglist[irun]['is_rerun']])\
                                                                         == 0 \
                           and len(newfinishedruns[irun]) > 0:
                            print('combined ',
                                finishedruns[offset+arglist[irun]['is_rerun']],
                                newfinishedruns[irun])
                        finishedruns[offset+arglist[irun]['is_rerun']] = \
                            np.append(
                                finishedruns[offset+arglist[irun]['is_rerun']],
                                newfinishedruns[irun])
                    else:
                        finishedruns[offset+irun] = newfinishedruns[irun]
                    newroots += len(newfinishedruns[irun])
            rungrid += newrungrid
            # Broadcast to all procs
            newroots = self.comm.bcast(newroots, root=0)
            finishedruns = self.comm.bcast(finishedruns, root=0)
            # Test validity of mapping
            for key in rungrid.ravel():
                get_if(finishedruns, key)
            # These are not obvious - Python sometimes gives a reference,
            # sometimes a copy.
            self.grids[-1]['runs'] = rungrid
            self.grids[-1]['results'] = finishedruns
        return newroots

    def to_hdf5(self):
        if not self.root:
            return
        print('Writing ', len(self.grids), 'grids to hdf5')
        with h5py.File(self.datafile_hdf5, 'w') as h5f:
            grp = h5f.create_group(self.batchname)
            grp.attrs['run_walltime'] = time.time() - self.wall_start
            for i, grid in enumerate(self.grids):
                self.write_grid(i, grid, h5f)
            h5f.close()

    def write_grid(self, igrid, grid, h5f):
        # Find depth for eigenvalue grid
        maxmult = 1
        for ir in grid['runs'].ravel():
            if ir >= 0:
                neigen = len(grid['results'][ir])
                if neigen > maxmult:
                    maxmult = neigen

        Kxgridout = np.zeros_like(grid['Kx'])
        Kzgridout = np.zeros_like(grid['Kz'])
        resultshape = list(grid['runs'].shape) + [maxmult]
        realpart = np.zeros(resultshape)
        imagpart = np.zeros(resultshape)
        error = -1*np.ones_like(grid['runs'], dtype=np.int)
        for (iz, ix), Kz in np.ndenumerate(grid['Kz']):
            Kxgridout[iz, ix] = grid['Kx'][iz, ix]
            Kzgridout[iz, ix] = grid['Kz'][iz, ix]
            roots = get_if(grid['results'], grid['runs'][iz, ix])
            error[iz, ix] = len(roots)
            if roots:
                for ir, root in enumerate(roots):
                    realpart[iz, ix, ir] = root.real
                    imagpart[iz, ix, ir] = root.imag

        grp = h5f.create_group(self.batchname+'/'+str(igrid))
        grp.attrs['stokes_range'] = \
            self.baseargs['__init__']['stokes_range']
        grp.attrs['dust_to_gas_ratio'] = \
            self.baseargs['__init__']['dust_to_gas_ratio']
        if 'viscous_alpha' in self.baseargs['calculate']:
            grp.attrs['viscous_alpha'] = \
                self.baseargs['calculate']['viscous_alpha']
        grp.attrs['real_range'] = self.baseargs['__init__']['real_range']
        grp.attrs['imag_range'] = self.baseargs['__init__']['imag_range']
        dset = grp.create_dataset('Kxgrid', data=Kxgridout)
        dset = grp.create_dataset('Kzgrid', data=Kzgridout)
        dset = grp.create_dataset('root_real', data=realpart)
        dset = grp.create_dataset('root_imag', data=imagpart)
        dset = grp.create_dataset('error', data=error)
