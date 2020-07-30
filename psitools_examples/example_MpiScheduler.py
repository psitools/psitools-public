# Demmo when run as MPI, this does a grid of calculations.
# To run computations
# PYTHONPATH=~/vc/dustcontinuum/ \
#   mpirun-mpich-mp -np 5 python3 -u test_MpiScheduler.py compute
#
# To collect tresults to a HDF5 file
# PYTHONPATH=~/vc/dustcontinuum/ \
#   python3  test_MpiScheduler.py compute

import numpy as np
#import psitools
import psitools.direct_mpi
from mpi4py import MPI
import time
import pickle
import sys
import h5py
import os
import errno

batchname = 'test_MPIScheduler'
database_hdf5 = batchname+'.hdf5'
prefix = 'test_outputs'

# arglist is a list of dictionaries, which are themselves
# the arguments for a Converger class
tsint = [1e-8, 0.01]
epstot = 0.5
beta = -3.5
alpha = 0.0
refine = 3
wall_start = time.time()
wall_limit_total = 70*60*60

# set up a wavenumber space grid
Kxaxis = np.logspace(-1, 3, 4)
Kzaxis = np.logspace(-1, 3, 4)
Kxgrid, Kzgrid = np.meshgrid(Kxaxis, Kzaxis)
ks = list(zip(Kxgrid.flatten(), Kzgrid.flatten()))

# Build a list with each combination of parameters
arglist = []
for k in ks:
    arglist.append({'tsint': tsint, 'epstot': epstot, 'beta': beta,
                    'Kx': k[0], 'Kz': k[1], 'refine': refine,
                    'gridding': 'chebyshevroots', 'alpha': alpha})


def collect_results():
    """Problem -specific callback to handle what to do once all the converger
       pickle files are on disk.
       This callback can be customized as needed,
       here we use a standard procedure
    """
    psitools.direct_mpi.collect_grid_powerlaw(arglist, Kxgrid, Kzgrid, Kxaxis,
                                         Kzaxis, batchname, database_hdf5)


# admittadly, detecting if this is being run as a script is redundant.
if __name__ == '__main__':
    # Need to specify here which dust distribution to run.
    # This maps to which Converger class to use.
    psitools.direct_mpi.runarglist(batchname, 'powerlaw', wall_limit_total, arglist,
                              collect_results)
