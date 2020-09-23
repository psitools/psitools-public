#
#  Module for direct solver tau_s gridding routines and definitions
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
# Who: Colin
import numpy as np
import numpy.linalg

# Maybe all this should be in a gridding module
gridmap = {'linear': 1, 'logarithmic': 2, 'chebyshev': 3,
           'logchebyshev': 4, 'chebyshevroots': 5,
           'logchebyshevroots': 6}


def chebgl(tsint, ntaus):
    '''Chebyshev Gauss-Lobatto grid
       Boyd (A.19)
    '''
    i = np.arange(ntaus)
    taus = (-0.5*np.cos(np.pi*i/(ntaus-1))+0.5) \
           * (tsint[1]-tsint[0]) + tsint[0]
    return taus


def chebgc(tsint, ntaus):
    '''Chebyshev Gauss-Chebyshev grid
       "Roots" grid - does not contain endpoints of the domain,
       which can be nice if there's a singularity at endpoint.
       Boyd (A.18)
    '''
    i = np.arange(ntaus)+1
    taus = (-0.5*np.cos((2*i-1)*np.pi/(2.0*ntaus))+0.5) \
           * (tsint[1]-tsint[0]) + tsint[0]
    return taus


def get_gridding(gridding, tsint, ntaus):
    """Utility used to generate direct solver grids in a standard way.

    Args
    gridding : Name from gridmap
    tsint : interval to grid
    ntaus : number of grid points
    """
    if gridding == gridmap['linear']:
        taus = np.linspace(tsint[0], tsint[1], ntaus)
    elif gridding == gridmap['logarithmic']:
        taus = np.logspace(np.log(tsint[0]),
                           np.log(tsint[1]), ntaus, base=np.e)
    elif gridding == gridmap['logchebyshev']:
        logint = (np.log(tsint[0]), np.log(tsint[1]))
        logtaus = chebgl(logint, ntaus)
        taus = np.exp(logtaus)
    elif gridding == gridmap['chebyshev']:
        taus = chebgl(tsint, ntaus)
    elif gridding == gridmap['logchebyshevroots']:
        logint = (np.log(tsint[0]), np.log(tsint[1]))
        logtaus = chebgc(logint, ntaus)
        taus = np.exp(logtaus)
    elif gridding == gridmap['chebyshevroots']:
        taus = chebgc(tsint, ntaus)
    else:
        raise ValueError('Unknown gridding in get_gridding', gridding)
    return taus
