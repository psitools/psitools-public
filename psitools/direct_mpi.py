#
# Module for the MPI parallel running of psitools.direct Convergers
#
import numpy as np
import numpy.linalg
import time
import sys
import errno
import pickle
import os.path
from mpi4py import MPI
import collections
import resource
import multiprocessing
import h5py

from . import direct


class MpiScheduler:
    """ Execute a set of AitkenConverger runs with a list of parameters
        on MPI tasks"""

    def __init__(self, arglist, wall_start, wall_limit_total,
                 disttype='powerlaw'):
        """__init__ calls everything, this is execute-on-instantiate
           arglist is a list of dictionaries, giving arguments for Converger"""
        self.exitFlag = False
        self.comm = MPI.COMM_WORLD
        assert self.comm.size > 1
        self.rank = self.comm.Get_rank()
        self.nranks = self.comm.Get_size()
        self.wall_start = wall_start
        self.wall_limit_total = wall_limit_total
        if disttype == 'powerlaw':
            self.Converger = direct.Converger
        elif disttype == 'lognormal':
            self.Converger = direct.ConvergerLogNormal
        elif disttype == 'powerbump':
            self.Converger = direct.ConvergerPowerBump
        else:
            raise ValueError('Unknown disttype '+disttype)
        if self.rank == 0:
            self.masterprocess(arglist)
        else:
            self.slaveprocess(arglist)
         MPI.Finalize()

    def masterprocess(self, campaign):
        """This is the master process, make a work list and send out
           assignments, do not work.
        """

        print('Master Process rank ', self.rank, flush=True)
        am = collections.deque(range(0, len(campaign)))
        finishedruns = []
        waiting = []
        for dest in range(1, self.nranks):
            if len(am) > 0:
                torun = am.popleft()
                self.comm.send(['run', torun], dest=dest, tag=1)
            else:
                self.comm.send(['wait'], dest=dest, tag=1)

        # go into a loop sending out and doing work
        exitFlag = False
        while (time.time() < self.wall_start + self.wall_limit_total
               and not exitFlag):
            status = MPI.Status()
            s = self.comm.probe(source=MPI.ANY_SOURCE, tag=2, status=status)
            finished = self.comm.recv(source=status.source, tag=2)
            if finished[0] == 'finished':
                finishedruns.append(finished[1])
            elif finished[0] == 'running':
                # add this run to back of queue
                am.append(finished[1])
            elif finished[0] == 'waiting':
                if status.source not in waiting:
                    waiting.append(status.source)
            else:
                print('Error, MpiScheduler unknown status message ', finished)
            # reply address
            dest = status.source
            # send work to process, or tell it to wait (only once!)
            if len(am) > 0:
                torun = am.popleft()
                self.comm.send(['run', torun], dest=dest, tag=1)
            elif dest not in waiting:
                # important not to send wait twice, a waiting dest
                # rank just needs an exit
                print('Rank {:d} commanded to wait'.format(dest), flush=True)
                self.comm.send(['wait'], dest=dest, tag=1)

            if (len(am) < 10):
                print('***Master unassigned list is now ',
                      list(am), flush=True)
            else:
                print('***Master unassigned list has {:d} entries.'
                      .format(len(am)), flush=True)
            if len(am) == 0 and len(waiting) == self.nranks-1:
                exitFlag = True

        # in this Async mode we need to make
        # sure to send this to slaves waiting
        # on the recv before exiting
        for dest in range(1, self.nranks):
            # we never track this, just let it die
            self.comm.isend(['exit'], dest=dest, tag=1)
        print('Exit rank ', self.rank, flush=True)

    def slaveprocess(self, campaign):
        """Execution of a MPI slave process, waits for commands from master,
           executes chunks of work from campaign"""
        print('Slave Process rank ', self.rank, flush=True)

        exitFlag = False
        while (time.time() < self.wall_start + self.wall_limit_total
               and not exitFlag):
            # post a blocking recv, to wait for an assignment
            # or exit from rank==0
            cmd = self.comm.recv(source=0, tag=1)
            if cmd[0] == 'run':
                print('Rank ', self.rank, ' args ', campaign[cmd[1]])
                p = multiprocessing.Process(target=self.runcompute,
                                            args=(campaign[cmd[1]],))
                p.start()
                p.join()
                # These are all non-blocking so we can get interrupted by a
                # shutdown if needed
                request = self.comm.isend(['finished', cmd[1]], dest=0, tag=2)
            elif cmd[0] == 'exit':
                exitFlag = True
            else:
                request = self.comm.isend(['waiting'], dest=0, tag=2)
            if self.rank == 1:
                print('peak memory usage ',
                      resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e3)
                print('unshared memory size ',
                      resource.getrusage(resource.RUSAGE_SELF).ru_idrss/1e3)
                print('unshared stack size ',
                      resource.getrusage(resource.RUSAGE_SELF).ru_isrss/1e3)

        print('Exit rank ', self.rank, flush=True)

    def runcompute(self, args):
        ait = self.Converger(**args)
        ait.runcompute()
        del(ait)


def runarglist(basename, disttype, wall_limit_total, arglist, collector):
    """ When running from the command line ans an MPI task,
        this handles commands """
    wall_start = time.time()
    if len(sys.argv) < 2:
        print('Need command')
        exit()

    if sys.argv[1] == 'compute':
        MpiScheduler(arglist, wall_start, wall_limit_total,
                     disttype=disttype)
    elif sys.argv[1] == 'recompute':
        with open(basename+'_reruns.txt', 'r') as rerunfile:
            content = rerunfile.readlines()
            content = [int(x.strip()) for x in content]
        arglist = [arglist[i] for i in content]
        MpiScheduler(arglist, wall_start, wall_limit_total,
                     disttype=disttype)
    elif sys.argv[1] == 'collect':
        collector()
    elif sys.argv[1] == 'clean':
        clean_results(arglist, disttype)
    else:
        print('Unknown command ', sys.argv[1])


def clean_results(arglist, disttype):
    """Utility for command line execution, clean up pickle files
       from Converger"""
    for args in arglist:
        if disttype == 'powerlaw':
            ait = direct.Converger(**args)
        elif disttype == 'lognormal':
            ait = direct.ConvergerLogNormal(**args)
        else:
            raise ValueError('Unknown disttype '+disttype)
        try:
            os.remove(ait.picklename)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise


def collect_grid_powerlaw(arglist, Kxgrid, Kzgrid, Kxaxis, Kzaxis, batchname,
                          database_hdf5):
    """Grid at varying Kz Kx only, powerlaw dust distribution"""

    tsint = arglist[0]['tsint']
    epstot = arglist[0]['epstot']
    beta = arglist[0]['beta']
    refine = arglist[0]['refine']

    # Read in the outputs
    fastesteigens = np.zeros([refine] + list(Kxgrid.shape), dtype=np.complex)
    backfastesteigens = np.zeros([refine]
                        + list(Kxgrid.shape), dtype=np.complex)
    computetimes = np.zeros([refine] + list(Kxgrid.shape))
    kxgridout = np.zeros(list(Kxgrid.shape))
    kzgridout = np.zeros(list(Kzgrid.shape))

    flatindex = 0
    for iz, kz in enumerate(Kzaxis):
        for ix, kx in enumerate(Kxaxis):
            ait = direct.Converger(**arglist[flatindex])
            try:
                with open(ait.picklename, 'rb') as picklefile:
                    ait = pickle.load(picklefile)
                    fastesteigens[:, iz, ix] = ait.fastesteigens
                    trace, backfastesteigens[:, iz, ix] = \
                        ait.backtraceeigen(ait.fastesteigens[-1],
                                           len(ait.fastesteigens)-1)
                    computetimes[:, iz, ix] = ait.computetimes
                    kxgridout[iz, ix] = arglist[flatindex]['Kx']
                    kzgridout[iz, ix] = arglist[flatindex]['Kz']
            except FileNotFoundError as fe:
                print(flatindex)
            del(ait)
            flatindex += 1

    with h5py.File(database_hdf5, 'w') as h5f:
        grp = h5f.create_group(batchname)
        dset = grp.create_dataset('Kxaxis', data=Kxaxis)
        dset = grp.create_dataset('Kzaxis', data=Kzaxis)
        dset = grp.create_dataset('fastesteigens', data=fastesteigens)
        dset = grp.create_dataset('backfastesteigens', data=backfastesteigens)
        dset = grp.create_dataset('computetimes', data=computetimes)
        dset = grp.create_dataset('Kxgrid', data=kxgridout)
        dset = grp.create_dataset('Kzgrid', data=kzgridout)
        grp.attrs['tsint'] = tsint
        grp.attrs['epstot'] = epstot
        grp.attrs['beta'] = beta
        h5f.close()


def collect_grid_powerbump(arglist, Kxgrid, Kzgrid, Kxaxis, Kzaxis, batchname,
                           database_hdf5):
    """Grid at varying Kz Kx only, powerbump dust distribution"""

    tsint = arglist[0]['tsint']
    epstot = arglist[0]['epstot']
    beta = arglist[0]['beta']
    aL = arglist[0]['aL']
    aP = arglist[0]['aP']
    bumpfac = arglist[0]['bumpfac']
    refine = arglist[0]['refine']

    # Read in the outputs
    fastesteigens = np.zeros([refine] + list(Kxgrid.shape), dtype=np.complex)
    backfastesteigens = np.zeros([refine] + list(Kxgrid.shape),
                                 dtype=np.complex)
    computetimes = np.zeros([refine] + list(Kxgrid.shape))
    kxgridout = np.zeros(list(Kxgrid.shape))
    kzgridout = np.zeros(list(Kzgrid.shape))

    flatindex = 0
    for iz, kz in enumerate(Kzaxis):
        for ix, kx in enumerate(Kxaxis):
            ait = direct.ConvergerPowerBump(**arglist[flatindex])
            try:
                with open(ait.picklename, 'rb') as picklefile:
                    ait = pickle.load(picklefile)
                    fastesteigens[:, iz, ix] = ait.fastesteigens
                    trace, backfastesteigens[:, iz, ix] = \
                        ait.backtraceeigen(ait.fastesteigens[-1],
                                           len(ait.fastesteigens)-1)
                    computetimes[:, iz, ix] = ait.computetimes
                    kxgridout[iz, ix] = arglist[flatindex]['Kx']
                    kzgridout[iz, ix] = arglist[flatindex]['Kz']
            except FileNotFoundError as fe:
                print(flatindex)
            del(ait)
            flatindex += 1

    with h5py.File(database_hdf5, 'w') as h5f:
        grp = h5f.create_group(batchname)
        dset = grp.create_dataset('Kxaxis', data=Kxaxis)
        dset = grp.create_dataset('Kzaxis', data=Kzaxis)
        dset = grp.create_dataset('fastesteigens', data=fastesteigens)
        dset = grp.create_dataset('backfastesteigens', data=backfastesteigens)
        dset = grp.create_dataset('computetimes', data=computetimes)
        dset = grp.create_dataset('Kxgrid', data=kxgridout)
        dset = grp.create_dataset('Kzgrid', data=kzgridout)
        grp.attrs['tsint'] = tsint
        grp.attrs['epstot'] = epstot
        grp.attrs['beta'] = beta
        grp.attrs['aL'] = aL
        grp.attrs['aP'] = aP
        grp.attrs['bumpfac'] = bumpfac
        h5f.close()
