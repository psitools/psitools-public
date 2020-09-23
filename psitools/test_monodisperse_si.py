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
import numpy.testing as npt
from . import monodisperse_si
import pytest

# Tolerance for tests
rtol = 1e-7


@pytest.mark.mpi_skip()
def test_mSI_0():
    mSI = monodisperse_si.MonodisperseSISolver(taus=0.1, epsilon=3.0)
    mSI.build_system_matrix(Kx=10.0, Kz=20.0)
    mSI.solve_eigen()
    ans = mSI.get_fastest_growth()

    print(ans)
    ref = 0.2918648213134704+0.021006933423273044j
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
def test_mSI_1():
    mSI = monodisperse_si.MonodisperseSISolver(taus=0.1, epsilon=3.0)
    mSI.build_system_matrix(Kx=10.0, Kz=20.0)
    mSI.add_turbulence(1e-6)
    mSI.solve_eigen()
    ans = mSI.get_fastest_growth()

    print(ans)
    ref = 0.6717245684834644+0.000621412365386553j
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
def test_mSI_2():
    mSI = monodisperse_si.MonodisperseSISolver(taus=0.1, epsilon=3.0)
    mSI.build_system_matrix(Kx=10.0, Kz=20.0)
    mSI.add_turbulence(1e-6)
    mSI.add_dust_pressure()
    mSI.solve_eigen()
    ans = mSI.get_fastest_growth()

    print(ans)
    ref = 0.6883764787241001+0.015856646724126157j
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)
