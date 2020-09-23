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
import numpy.testing as npt
from .terminalvelocitysolver import TerminalVelocitySolver
import pytest

# Tolerance for tests
rtol = 1e-10


@pytest.mark.mpi_skip()
def test_0():
    tau_min = np.logspace(-8, -2, 100)
    mode = TerminalVelocitySolver(dust_gas_ratio=3,
                                  wave_number_x=46,
                                  wave_number_z=1,
                                  minimum_stopping_time=tau_min,
                                  maximum_stopping_time=0.01,
                                  power_law_exponent_size_distribution=-3.5)
    w = mode.find_roots()

    ans = w[-3:]
    print('Last 3 roots')
    for an in ans:
        print(an)

    ref = np.array((0.01000783281810628+0.024248160108983886j,
                    0.009256948351079859+0.024541272898239125j,
                    0.008859085339836027+0.024878670596489044j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)
