#
#  Just run pytest in the repo, and this will get picked up and run.
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
import numpy.testing as npt
import pytest
from . import direct

# Tolerance for tests
# Needs to be rather loose as multithreaded BLAS results vary
rtol = 5e-6


@pytest.mark.mpi_skip()
def run_PL(testargs):
    ait = direct.Converger(**testargs)
    ait.runcompute()
    print('-------------------------------------')
    print(testargs)
    print('Fastest Eigenvalues')
    for val in ait.fastesteigens:
        print(val)
    return ait.fastesteigens


@pytest.mark.mpi_skip()
def run_PB(testargs):
    ait = direct.ConvergerPowerBump(**testargs)
    ait.runcompute()
    print('-------------------------------------')
    print(testargs)
    print('Fastest Eigenvalues')
    for val in ait.fastesteigens:
        print(val)
    return ait.fastesteigens


@pytest.mark.mpi_skip()
def run_PBT(testargs):
    ait = direct.ConvergerPowerBumpTail(**testargs)
    ait.runcompute()
    print('-------------------------------------')
    print(testargs)
    print('Fastest Eigenvalues')
    for val in ait.fastesteigens:
        print(val)
    return ait.fastesteigens


@pytest.mark.mpi_skip()
def run_LN(testargs):
    ait = direct.ConvergerLogNormal(**testargs)
    ait.runcompute()
    print('-------------------------------------')
    print(testargs)
    print('Fastest Eigenvalues')
    for val in ait.fastesteigens:
        print(val)
    return ait.fastesteigens


@pytest.mark.mpi_skip()
@pytest.mark.timeout(60)
def test_doesitrun():
    ntaus = 25
    tsint = (1e-8, 1e-2)
    epstot = 3.0
    beta = -3.5
    alpha = 1e-8
    sigdiff = 1e-9
    dust_pressure = True
    Kx = Kz = 1e3
    i = np.arange(ntaus)+1
    taus = (-0.5*np.cos(np.pi*(2*i-1)/(2.0*ntaus))+0.5) \
           * (tsint[1]-tsint[0]) + tsint[0]
    ss = direct.StreamingSolver(taus,
                                        epstot=epstot,
                                        beta=beta,
                                        alpha=alpha,
                                        dust_pressure=dust_pressure,
                                        sigdiff=sigdiff)
    ss.build_system_matrix(Kx=Kx, Kz=Kz)
    ss.solve_eigen()


@pytest.mark.mpi_skip()
@pytest.mark.slow()
@pytest.mark.timeout(60)
def test_PB_0(tmpdir):
    ans = run_PB({
            'tsint': (1e-08, 1.56),
            'epstot': 3,
            'beta': -3.5,
            'aL': 2.0/3.0*1.0,
            'aP': 1.0,
            'bumpfac': 2.0,
            'Kx': 4.0,
            'Kz': 100.0,
            'refine': 3,
            'gridding': 'chebyshevroots',
            'prefix': tmpdir,
            'alpha': None})
    ref = np.array((-0.04298854605814622+0.13764347038117714j,
                    -0.04401630203296608+0.13752657929813472j,
                    -0.04454892638085971+0.1374699433910088j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.slow()
@pytest.mark.timeout(60)
def test_PB_1(tmpdir):
    ans = run_PB({
            'tsint': (1e-08, 1.56),
            'epstot': 3,
            'beta': -3.5,
            'aL': 2.0/3.0*1.0,
            'aP': 1.0,
            'bumpfac': 2.0,
            'Kx': 4.0,
            'Kz': 100.0,
            'refine': 3,
            'gridding': 'chebyshevroots',
            'prefix': tmpdir,
            'alpha': 4e-8,
            'dust_pressure': True})
    ref = np.array(
                   (0.13348830950613355-0.05168309862307213j,
                    0.13242757784498815-0.05197300068236282j,
                    0.13188176480295802-0.052123120488923765j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.slow()
@pytest.mark.timeout(60)
def test_PB_2(tmpdir):
    ans = run_PB({
             'tsint': (1e-08, 1.56),
             'epstot': 3,
             'beta': -3.5,
             'aL': 2.0/3.0*1.0,
             'aP': 1.0,
             'bumpfac': 2.0,
             'Kx': 4.0,
             'Kz': 100.0,
             'refine': 3,
             'gridding': 'logarithmic',
             'prefix': tmpdir,
             'alpha': 4e-8})
    ref = np.array(
                   (-0.0024531728633354326-0.010686752395676097j,
                    -0.0014217405603043848-0.008240758021806682j,
                    -0.0014180550135648997-0.007518473660759821j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.timeout(60)
def test_PB_3(tmpdir):
    ans = run_PB({
            'tsint': (1e-08, 1.56),
            'epstot': 3,
            'beta': -3.5,
            'aL': 2.0/3.0*1.0,
            'aP': 1.0,
            'bumpfac': 2.0,
            'Kx': 4.0,
            'Kz': 100.0,
            'refine': 1,
            'gridding': 'chebyshevroots',
            'prefix': tmpdir,
            'alpha': None})
    ref = np.array((-0.04298854605814622+0.13764347038117714j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.slow()
@pytest.mark.timeout(60)
def test_PBT_0(tmpdir):
    ans = run_PBT({
             'tsint': (1e-08, 1.56),
             'epstot': 3,
             'beta': -3.5,
             'aL': 2.0/3.0*1.0,
             'aP': 1.0,
             'aBT': 1.0e-4,
             'bumpfac': 2.0,
             'Kx': 4.0,
             'Kz': 100.0,
             'refine': 3,
             'gridding': 'chebyshevroots',
             'prefix': tmpdir,
             'alpha': None})
    ref = np.array(
                   (-0.04153265004675379+0.13846049287817908j,
                    -0.041966635201637514+0.13811256698525176j,
                    -0.0418667083429648+0.13817651254264796j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.slow()
@pytest.mark.timeout(60)
def test_PBT_1(tmpdir):
    ans = run_PBT({
             'tsint': (1e-08, 1.56),
             'epstot': 3,
             'beta': -3.5,
             'aL': 2.0/3.0*1.0,
             'aP': 1.0,
             'aBT': 1.0e-4,
             'bumpfac': 2.0,
             'Kx': 4.0,
             'Kz': 100.0,
             'refine': 3,
             'gridding': 'logarithmic',
             'prefix': tmpdir,
             'alpha': 4e-8})
    ref = np.array(
                   (0.001106062161350275-0.009826073486222765j,
                    0.002129071217410124-0.007350562586222042j,
                    0.002131168782156639-0.00662131639196411j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.slow()
@pytest.mark.timeout(60)
def test_PL_0(tmpdir):
    # MRN tests
    ans = run_PL({
            'tsint': (1e-08, 1e-1),
            'epstot': 3.0,
            'beta': -3.5,
            'Kx': 80.0,
            'Kz': 100.0,
            'refine': 3,
            'gridding': 'chebyshevroots',
            'prefix': tmpdir,
            'alpha': 4e-8,
            'dust_pressure': True})
    ref = np.array(
                   (0.605001947246457+0.01855421363780353j,
                    0.6026448174624307+0.018214074745258046j,
                    0.6014473718398458+0.018048063756147426j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.slow()
@pytest.mark.timeout(60)
def test_PL_1(tmpdir):
    ans = run_PL({
            'tsint': (1e-08, 1e-1),
            'epstot': 3.0,
            'beta': -3.5,
            'Kx': 80.0,
            'Kz': 100.0,
            'refine': 3,
            'gridding': 'chebyshevroots',
            'prefix': tmpdir,
            'alpha': None})
    ref = np.array((0.5512363292041821+0.11242857992948893j,
                    0.5489690582185277+0.11216071905699404j,
                    0.5478173144269354+0.11203395665778133j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
@pytest.mark.slow()
@pytest.mark.timeout(60)
def test_LN_0(tmpdir):
    ans = run_LN({
               'tsint': (1e-08, 1e-1),
               'epstot': 3.0,
               'sigma': 1e-2,
               'peak': 8e-2,
               'Kx': 100.0,
               'Kz': 100.0,
               'refine': 3,
               'gridding': 'chebyshevroots',
               'prefix': tmpdir,
               'alpha': 1e-7})
    ref = np.array(
                   (0.40950662474594923+0.2743694562314839j,
                    0.40944101911374947+0.2743957294399602j,
                    0.409424614329867+0.2744022809946226j))
    npt.assert_allclose(ans.real, ref.real, rtol=rtol)
    npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)
