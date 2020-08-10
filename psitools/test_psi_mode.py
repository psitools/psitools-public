import numpy as np
import numpy.testing as npt
from .power_bump import PowerBump, PowerBumpTail
from .sizedensity import SizeDensity
from .psi_mode import PSIMode
from .tanhsinh import TanhSinh
import pytest

# Tolerance for tests
rtol = 1e-7


@pytest.mark.mpi_skip()
def test_psi_mode_0():
    np.random.seed(0)
    pm = PSIMode(dust_to_gas_ratio=3,
             stokes_range=[1.0e-8, 0.1],
             real_range=[-2, 2],
             imag_range=[1.0e-8, 1],
             single_size_flag=True,
             tanhsinh_integrator=TanhSinh())
    roots = pm.calculate(wave_number_x=30,
                         wave_number_z=30)
    print('roots')
    for root in roots:
        print(root)
    refs = [0.34801869149969916+0.4190301951732903j]
    for ans, ref in zip(roots, refs):
        npt.assert_allclose(ans.real, ref.real, rtol=rtol)
        npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
def test_psi_mode_1():
    np.random.seed(0)
    pm = PSIMode(dust_to_gas_ratio=3,
             stokes_range=[1.0e-8, 0.01],
             real_range=[-2, 2],
             imag_range=[1.0e-8, 1])
    roots = pm.calculate(wave_number_x=10,
                         wave_number_z=1000)
    print('roots')
    for root in roots:
        print(root)
    refs = [-1.0062579942546936+1.3209787635623396e-05j]
    for ans, ref in zip(roots, refs):
        npt.assert_allclose(ans.real, ref.real, rtol=rtol)
        npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
def test_psi_mode_2():
    np.random.seed(0)
    pm = PSIMode(dust_to_gas_ratio=3,
                 stokes_range=[1.0e-8, 1],
                 real_range=[-2, 2],
                 imag_range=[1.0e-8, 1])
    roots = pm.calculate(wave_number_x=0.1,
                         wave_number_z=10)
    print('roots')
    for root in roots:
        print(root)
    refs = [-1.0048799539684967+0.0008946414712527288j]
    for ans, ref in zip(roots, refs):
        npt.assert_allclose(ans.real, ref.real, rtol=rtol)
        npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
def test_psi_mode_3():
    amin = 1.0e-8
    aL = 2.0/3.0*0.1
    aP = 0.1
    aR = 0.156
    bumpfac = 2.0
    epstot = 10.0
    pb = PowerBump(amin=amin, aP=aP, aL=aL, aR=aR, bumpfac=bumpfac)
    sd = SizeDensity(pb.sigma0, [amin, aR])
    pole = pb.get_discontinuity()
    sd.poles = [pole]

    np.random.seed(0)
    pm = PSIMode(dust_to_gas_ratio=epstot,
                 stokes_range=[amin, aR],
                 real_range=[-2, 2],
                 imag_range=[1.0e-8, 1],
                 n_sample=15,
                 max_zoom_domains=1,
                 verbose_flag=False,
                 single_size_flag=False,
                 size_density=sd,
                 tanhsinh_integrator=TanhSinh())

    Kx = 200
    Kz = 1000
    roots = pm.calculate(wave_number_x=Kx, wave_number_z=Kz)

    print('roots')
    for root in roots:
        print(root)
    refs = [0.36109234399742074+0.6184094307832038j]
    for ans, ref in zip(roots, refs):
        npt.assert_allclose(ans.real, ref.real, rtol=rtol)
        npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
def test_psi_mode_4():
    amin = 1.0e-8
    aBT = 1e-6
    aL = 2.0/3.0*0.1
    aP = 0.1
    aR = 0.156
    bumpfac = 2.0
    epstot = 10.0
    pb = PowerBumpTail(amin=amin, aBT=aBT,
                       aP=aP, aL=aL, aR=aR, bumpfac=bumpfac)
    sd = SizeDensity(pb.sigma0, [amin, aR])
    pole = pb.get_discontinuity()
    sd.poles = [pole]

    np.random.seed(0)
    pm = PSIMode(dust_to_gas_ratio=epstot,
                 stokes_range=[amin, aR],
                 real_range=[-2, 2],
                 imag_range=[1.0e-8, 1],
                 n_sample=15,
                 max_zoom_domains=1,
                 verbose_flag=False,
                 single_size_flag=False,
                 size_density=sd,
                 tanhsinh_integrator=TanhSinh())

    Kx = 200
    Kz = 1000
    roots = pm.calculate(wave_number_x=Kx, wave_number_z=Kz)

    print('roots')
    for root in roots:
        print(root)
    refs = [0.36165047122154015+0.619087269550498j]
    for ans, ref in zip(roots, refs):
        npt.assert_allclose(ans.real, ref.real, rtol=rtol)
        npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)


@pytest.mark.mpi_skip()
def test_psi_mode_5():
    np.random.seed(0)
    pm = PSIMode(dust_to_gas_ratio=3.0,
                 stokes_range=[1e-8, 0.1],
                 real_range=[-2, 2],
                 imag_range=[1.0e-8, 1],
                 n_sample=15,
                 max_zoom_domains=1,
                 verbose_flag=False,
                 single_size_flag=False)

    Kx = 80.0
    Kz = 100.0
    roots = pm.calculate(wave_number_x=Kx, wave_number_z=Kz,
                         viscous_alpha=4e-8)

    print('roots')
    for root in roots:
        print(root)
    refs = [0.5682740647046338+0.021314557033861297j]
    for ans, ref in zip(roots, refs):
        npt.assert_allclose(ans.real, ref.real, rtol=rtol)
        npt.assert_allclose(ans.imag, ref.imag, rtol=rtol)
