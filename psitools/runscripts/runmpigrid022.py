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

batchname = 'grid022'

tsint = [1e-2, 1.0]
epstot = 1.0
beta = 0.0
refine = 5
wall_start = time.time()
wall_limit_total = 70*60*60

database_hdf5 = batchname+'.hdf5'


kxaxis=np.logspace(-1,3,64)
kzaxis=np.logspace(-1,3,64)
kxgrid, kzgrid = np.meshgrid(kxaxis,kzaxis)
ks = list(zip(kxgrid.flatten(), kzgrid.flatten()))
arglist = []
for k in ks:
    arglist.append({'tsint':tsint, 'epstot':epstot, 'beta':beta, 'kx':k[0], 'kz':k[1], 'refine':refine})

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
            try:
                with open(ait.picklename,'rb') as picklefile:
                    ait = pickle.load(picklefile)
                    fastesteigens[:,ix,iz] = ait.fastesteigens
                    aitkengrowthrates[:,ix,iz] = ait.aitkengrowthrates
                    aitkenoscillate[:,ix,iz] = ait.aitkenoscillate
                    computetimes[:,ix,iz] = ait.computetimes
            except FileNotFoundError as fe:
                print(flatindex)
            del(ait)
            flatindex += 1

    with h5py.File(database_hdf5,'w') as h5f:
        grp = h5f.create_group(batchname)
        dset = grp.create_dataset('kxaxis', data=kxaxis)    
        dset = grp.create_dataset('kzaxis', data=kzaxis)    
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


