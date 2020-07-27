#
# a MPI based driver for running grids of PSIMode root finding calculations
#  see testPSIMode.py for example usage
#
import time
import numpy as np
import collections
from mpi4py import MPI
import multiprocessing
from .psi_mode import PSIMode


class MpiScheduler:
    """Execute a set of runs with a list of parameters on MPI tasks.
    """

    def __init__(self, wall_start, wall_limit_total, verbose=True):
        """__init__ calls everything, this is execute-on-instantiate
           arglist is a list of dictionaries, giving arguments """
        self.verbose = verbose
        self.exitFlag = False
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nranks = self.comm.Get_size()
        self.wall_start = wall_start
        self.wall_limit_total = wall_limit_total
        assert self.comm.size > 1

    def run(self, arglist):
        if self.rank == 0:
            return self.masterprocess(arglist)
        else:
            self.slaveprocess(arglist)
            return None  # so we know in caller this was a slave
        MPI.Finalize()

    def masterprocess(self, campaign):
        """This is the master process, make a work list and send out
           assignments, do not work.
        """

        if self.verbose:
            print('Master Process rank ', self.rank, flush=True)
        emptyflag = False
        am = collections.deque(range(0, len(campaign)))
        finishedruns = {}
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
                if self.verbose:
                    print(finished[1], 'result ', finished[2])
                finishedruns[finished[1]] = finished[2]
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
                # important not to send wait twice,
                # a waiting dest rank just needs an exit
                if self.verbose:
                    print('Rank {:d} commanded to wait'.format(dest),
                          flush=True)
                self.comm.send(['wait'], dest=dest, tag=1)

            if (len(am) < 10 and not emptyflag):
                print('***Master unassigned list is now ',
                      list(am), flush=True)
                if len(am) == 0:
                    emptyflag = True
            else:
                if self.verbose:
                    print('***Master unassigned list has {:d} entries.\n'
                          .format(len(am)), flush=True)
            if len(am) == 0 and len(waiting) == self.nranks-1:
                exitFlag = True

        # In this Async mode we need to make sure to send this to slaves
        # waiting on the recv before exiting.
        for dest in range(1, self.nranks):
            # we never track this, just let it die
            self.comm.isend(['exit'], dest=dest, tag=1)
        if self.verbose:
            print('Exit rank ', self.rank, flush=True)
        return finishedruns

    def slaveprocess(self, campaign):
        """Execution of a MPI slave process, waits for commands from master,
           executes chunks of work from campaign
        """
        if self.verbose:
            print('Slave Process rank ', self.rank, flush=True)

        # cache the object, need here to know none have been generated
        self.PSIModeArgs = {}

        exitFlag = False
        while (time.time() < self.wall_start + self.wall_limit_total
               and not exitFlag):
            # Post a blocking recv, to wait for
            # an assignment or exit from rank==0
            cmd = self.comm.recv(source=0, tag=1)
            if cmd[0] == 'run':
                if self.verbose:
                    print('Rank ', self.rank, ' args ', campaign[cmd[1]])
                maxroots = 128
                # array typecode l should mathc np.int32
                sharednroots = multiprocessing.RawArray('l', 1)
                sharedrootsreal = multiprocessing.RawArray('d', maxroots)
                sharedrootsimag = multiprocessing.RawArray('d', maxroots)
                p = multiprocessing.Process(target=self.runcompute,
                                            args=(campaign[cmd[1]],
                                                  sharednroots,
                                                  sharedrootsreal,
                                                  sharedrootsimag))
                p.start()
                p.join()
                print('got sharednroots', sharednroots[0])
                result = np.zeros(sharednroots[0], dtype=np.complex)
                result.real = sharedrootsreal[:sharednroots[0]]
                result.imag = sharedrootsimag[:sharednroots[0]]
                # These are all non-blocking so we can get interrupted
                # by a shutdown if needed.
                #result = self.runcompute(campaign[cmd[1]])
                request = self.comm.isend(['finished', cmd[1], result],
                                          dest=0, tag=2)
            elif cmd[0] == 'exit':
                exitFlag = True
            else:
                request = self.comm.isend(['waiting'], dest=0, tag=2)

        if self.verbose:
            print('Exit rank ', self.rank, flush=True)

    def runcompute(self, args, sharednroots, sharedrootsreal, sharedrootsimag):
        """The call to PSIMode."""
        if 'random_seed' in args:
            np.random.seed(args.pop('random_seed'))
        else:
            np.random.seed(0)

        PSIModeArgs = args['__init__']
        if not (PSIModeArgs == self.PSIModeArgs):
            self.pm = PSIMode(**PSIModeArgs)
            self.PSIModeArgs = PSIModeArgs
        roots = self.pm.calculate(**args['calculate'])
        nroots = np.frombuffer(sharednroots, dtype=np.int64)
        np.copyto(nroots, np.array([len(roots)], dtype=np.int64))
        if len(roots) > 0:
            realpart = np.frombuffer(sharedrootsreal, dtype=np.float64)
            np.copyto(realpart[:len(roots)], roots.real)
            imagpart = np.frombuffer(sharedrootsimag, dtype=np.float64)
            np.copyto(imagpart[:len(roots)], roots.imag)
        # return roots.copy()


class DispersionRelationMpiScheduler(MpiScheduler):
    """Class for computing  values of the dispersion relation.
    """

    def slaveprocess(self, campaign):
        """Execution of a MPI slave process, waits for commands from master,
           executes chunks of work from campaign
        """
        if self.verbose:
            print('Slave Process rank ', self.rank, flush=True)

        # cache the object, need here to know none have been generated
        self.PSIModeArgs = {}

        exitFlag = False
        while (time.time() < self.wall_start + self.wall_limit_total
               and not exitFlag):
            # Post a blocking recv, to wait for
            # an assignment or exit from rank==0
            cmd = self.comm.recv(source=0, tag=1)
            if cmd[0] == 'run':
                if self.verbose:
                    print('Rank ', self.rank, ' args ', campaign[cmd[1]])
                # Try to get away without multiprocessing.
                # Hopefully the garbage collector will work well enough
                result = self.runcompute(campaign[cmd[1]])
                request = self.comm.isend(['finished', cmd[1], result],
                                          dest=0, tag=2)
            elif cmd[0] == 'exit':
                exitFlag = True
            else:
                request = self.comm.isend(['waiting'], dest=0, tag=2)

        if self.verbose:
            print('Exit rank ', self.rank, flush=True)

    def runcompute(self, args):
        PSIModeArgs = args['__init__']
        if not (PSIModeArgs == self.PSIModeArgs):
            self.pm = PSIMode(**PSIModeArgs)
            self.PSIModeArgs = PSIModeArgs

        dispersion = self.pm.dispersion(**args['dispersion'])
        return dispersion
