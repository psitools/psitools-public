# when run as MPI, this does a grid of calculations. 
# 
import numpy as np
import streamingtools
from mpi4py import MPI
import time
import pickle
import sys
import h5py
import os
import errno

tsint = [1e-4, 1e-0]
epstot = 1.0
beta = -3.5
refine = 4
wall_start = time.time()
wall_limit_total = 70*60*60

database_hdf5 = 'grid010.hdf5'


kxaxis=np.logspace(-1,1,64*2)
kzaxis=np.logspace(-1,1,64*2)
kxgrid, kzgrid = np.meshgrid(kxaxis,kzaxis)
ks = list(zip(kxgrid.flatten(), kzgrid.flatten()))
arglist = []
for k in ks:
    arglist.append({'tsint':tsint, 'epstot':epstot, 'beta':beta, 'Kx':k[0], 'Kz':k[1], 'refine':refine})

def clean_results():
    for args in arglist:
        ait = streamingtools.AitkenConverger(**args)
        try:
            os.remove(ait.picklename)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

def collect_results():
    # Read in the outputs
    fastesteigens = np.zeros([refine] + list(kxgrid.shape), dtype=np.complex)
    aitkengrowthrates = np.zeros([refine-2] + list(kxgrid.shape))
    aitkenoscillate = np.zeros([refine-2] + list(kxgrid.shape))
    computetimes = np.zeros([refine] + list(kxgrid.shape))
   
    flatindex = 0
    for ix, kx in enumerate(kxaxis):
        for iz, kz in enumerate(kzaxis):
            ait = streamingtools.AitkenConverger(**arglist[flatindex])
            with open(ait.picklename,'rb') as picklefile:
                ait = pickle.load(picklefile)
                fastesteigens[:,ix,iz] = ait.fastesteigens
                aitkengrowthrates[:,ix,iz] = ait.aitkengrowthrates
                aitkenoscillate[:,ix,iz] = ait.aitkenoscillate
                computetimes[:,ix,iz] = ait.computetimes
            del(ait)
            flatindex += 1

    with h5py.File(database_hdf5,'w') as h5f:
        grp = h5f.create_group('grid010')
        dset = grp.create_dataset('Kxaxis', data=kxaxis)    
        dset = grp.create_dataset('Kzaxis', data=kzaxis)    
        dset = grp.create_dataset('fastesteigens', data=fastesteigens)    
        dset = grp.create_dataset('aitkengrowthrates', data=aitkengrowthrates)    
        dset = grp.create_dataset('aitkenoscillate', data=aitkenoscillate)    
        dset = grp.create_dataset('computetimes', data=computetimes)    
        grp.attrs['tsint'] = tsint
        grp.attrs['epstot'] = epstot
        grp.attrs['beta'] = beta
        h5f.close()


if __name__ == '__main__':
     
    if len(sys.argv) < 2:
        print('Need command')
        exit()
     
    if sys.argv[1] == 'compute':
        streamingtools.MpiScheduler(arglist, wall_start, wall_limit_total)
    elif sys.argv[1] == 'collect':
        collect_results()
    elif sys.argv[1] == 'clean':
        clean_results()
    else:
        print('Unknown command ',sys.argv[1])

