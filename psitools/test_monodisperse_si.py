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
