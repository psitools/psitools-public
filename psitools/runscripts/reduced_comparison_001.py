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

batchname = 'reduced_comparison_001'

#params from SJ
epstot = 0.2
fg = 1.0/(1.0+epstot)
tsmax = 0.1
kx = 0.1
kz = 0.1
beta = -3.5
prefix = 'reduced_comparison'

refine = 5
wall_start = time.time()
wall_limit_total = 70*60*60

database_hdf5 = batchname+'.hdf5'


tsmins = np.logspace(np.log10(4e-2),np.log10(0.0999),47)

arglist = []
for tsmin in tsmins:
    arglist.append({'tsint':(tsmin, tsmax), 'epstot':epstot, 'beta':beta, 'kx':kx, 'kz':kz, 'refine':refine, 'prefix':prefix})

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
    fastesteigens = np.zeros([refine] + list(tsmins.shape), dtype=np.complex)
    aitkengrowthrates = np.zeros([refine-2] + list(tsmins.shape))
    aitkenoscillate = np.zeros([refine-2] + list(tsmins.shape))
    computetimes = np.zeros([refine] + list(tsmins.shape))
   
    flatindex = 0
    for it, tsmin in enumerate(tsmins):
        ait = streamingtools.AitkenConverger(**arglist[flatindex])
        try:
            with open(ait.picklename,'rb') as picklefile:
                ait = pickle.load(picklefile)
                fastesteigens[:,it] = ait.fastesteigens
                aitkengrowthrates[:,it] = ait.aitkengrowthrates
                aitkenoscillate[:,it] = ait.aitkenoscillate
                computetimes[:,it] = ait.computetimes
        except FileNotFoundError as fe:
            print(flatindex)
        del(ait)
        flatindex += 1

    with h5py.File(database_hdf5,'w') as h5f:
        grp = h5f.create_group(batchname)
        dset = grp.create_dataset('tsmins', data=tsmins)    
        dset = grp.create_dataset('fastesteigens', data=fastesteigens)    
        dset = grp.create_dataset('aitkengrowthrates', data=aitkengrowthrates)    
        dset = grp.create_dataset('aitkenoscillate', data=aitkenoscillate)    
        dset = grp.create_dataset('computetimes', data=computetimes)    
        grp.attrs['kx'] = kx
        grp.attrs['kz'] = kz
        grp.attrs['tsmax'] = tsmax
        grp.attrs['epstot'] = epstot
        grp.attrs['distribution'] = 'powerlaw'
        grp.attrs['beta'] = beta
        h5f.close()


if __name__ == '__main__':
     
    if len(sys.argv) < 2:
        print('Need command')
        exit()
     
    if sys.argv[1] == 'compute':
        streamingtools.MpiScheduler(arglist, wall_start, wall_limit_total, disttype = 'powerlaw')
    elif sys.argv[1] == 'collect':
        collect_results()
    elif sys.argv[1] == 'clean':
        clean_results()
    else:
        print('Unknown command ',sys.argv[1])
