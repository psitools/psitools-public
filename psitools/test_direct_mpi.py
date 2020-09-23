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
# Who: Colin
import pytest
import numpy as np
import psitools.direct_mpi


# This really should have the min_size argument, but that seems to
# cause the test to not be collected.
@pytest.mark.mpi  # (min_size=2)
def test_direct_mpi_0(tmpdir):
    batchname = 'test_MPIScheduler'
    database_hdf5 = batchname+'.hdf5'
    prefix = tmpdir

    tsint = [1e-8, 0.01]
    epstot = 0.5
    beta = -3.5
    alpha = 0.0
    refine = 3
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
                        'gridding': 'chebyshevroots', 'alpha': alpha,
                        'prefix': prefix})

    def collect_results():
        psitools.direct_mpi.collect_grid_powerlaw(arglist, Kxgrid, Kzgrid,
                                      Kxaxis, Kzaxis, batchname, database_hdf5)

    psitools.direct_mpi.runarglist(batchname, 'powerlaw',
                                    wall_limit_total, arglist, collect_results)
