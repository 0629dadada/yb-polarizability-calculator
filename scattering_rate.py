#scattering rate of 1S0 and 3P1
import numpy as np
from sympy.physics.wigner import wigner_6j
from sympy import Rational
import math
import os
from datetime import datetime
import matplotlib.pyplot as plt


def scatterrate(lambda_nm, istate):
    
    c = 299792458
    h = 6.62607004e-34
    hbar = h/2/np.pi
    epsilon0 = 8.8541878128e-12
    a0 = 0.52917721067e-10
    au = (4 * np.pi * epsilon0 * a0**3)
    auh = (4 * np.pi * epsilon0 * a0**3) / h
    e = 1.602176634e-19           # C, elementary charge
    ea0 = e * a0                        # atomic unit of dipole moment (C*m)

    couplestate = {} #states allowed by dipole transition
    wavenumber = {}
    w = {} #angular frequency
    RME = {} #reduced matrix element
    RME_SI = {}
    Jk = {}
    Ji = {}
    LW = {}


    # (6s2)1S0: Thesis: "Interorbital spin exchange in a state-dependent optical
    # lattice" table B.1 + paper"Determination of static dipole polarizabilities of Yb atom"
    couplestate[1] = ['(6s6p)3P1', '(6s6p)1P1', '(7/2,5/2)o', '(6s7p)1P1', '(6s7p)3P1', '(6s8p)1P1', '(6s8p)3P1']
    wavenumber[1] = np.array([17992.00699, 25068.222, 28857.014, 40563.97, 38174.17, 44017.6, 43659.38])
    w[1] = 2 * np.pi * c * wavenumber[1] * 100  # angular frequency
    Jk[1] = np.ones_like(w[1], dtype=int)
    Ji[1] = 0
    RME[1] = np.array([0.543, 4.148, 2.04, 0.67, 0.08, 0.21, 0.014])
    RME_SI[1] = RME[1] * ea0
    LW[1] = (RME_SI[1]**2*w[1]**3/(3*np.pi*epsilon0*hbar*c**3))/(2*Jk[1]+1)
    
    # (6s6p)3P0: Thesis: "Interorbital spin exchange in a state-dependent optical
    # lattice" table B.1 + paper"Determination of static dipole polarizabilities of Yb atom"
    couplestate[2] = ['(5d6s)3D1', '(6s6d)3D1', '(6S7d)3D1', '(6s7s)3S1', '(6s8s)3S1', '(6p2)3P1', 'eff']
    wavenumber[2] = np.array([24489.102, 39808.72, 44311.38, 32694.692, 41615.04, 43805.42, 1 / (375 * 1e-7) + 17288.439 ]) - 17288.439
    w[2] = 2 * np.pi * c * wavenumber[2] * 100  # angular frequency
    Jk[2] = np.ones_like(w[2], dtype=int)
    Ji[2] = 0
    RME[2] = np.array([2.96, 1.79, 2.10, 1.95, 0.62, 1.97, 2.25])
    RME_SI[2] = RME[2] * ea0
    LW[2] = (RME_SI[2]**2*w[2]**3/(3*np.pi*epsilon0*hbar*c**3))/(2*Jk[2]+1)

    wlight = 2 * np.pi * c / (np.array(lambda_nm) * 1e-9)
    sc = np.zeros_like(wlight, dtype=float)
    if istate == 1:
        for k in range(len(Jk[istate])):
            sc += (-1)**(Ji[istate]+Jk[istate][k]+1)*(LW[istate][k]/2/((w[istate][k]-wlight)**2+(LW[istate][k]/2)**2)+LW[istate][k]/(2*(w[istate][k]+wlight)**2+(LW[istate][k]/2)**2))*RME_SI[istate][k]**2/3/hbar/(2*Ji[istate]+1)/(hbar*c*epsilon0)
            rate = sc
        return rate
    elif istate == 2:
        for k in range(len(Jk[istate])):
            sc += (-1)**(Ji[istate]+Jk[istate][k]+1)*(LW[istate][k]/2/((w[istate][k]-wlight)**2+(LW[istate][k]/2)**2)+LW[istate][k]/(2*(w[istate][k]+wlight)**2+(LW[istate][k]/2)**2))*RME_SI[istate][k]**2/3/hbar/(2*Ji[istate]+1)/(hbar*c*epsilon0)
            rate = sc
        return rate

def single_transition_scattering(Delta, Intensity, Gamma):
    """
    專門計算近共振、單一躍遷的飽和散射率
    """
    c = 299792458
    h = 6.62607004e-34
    hbar = h/2/np.pi
    epsilon0 = 8.8541878128e-12
    a0 = 0.52917721067e-10

    f = c / (555.8026*1e-9)
    f_new = f - Delta/(2*np.pi)
    lambda_detune = c/f_new
    τ = 1/Gamma
    Isat = np.pi*h*c/3/(555.8026*1e-9)**3 * Gamma
    s0 = Intensity / Isat
    # 完整的飽和 Lorentzian 公式
    R_sc = (Gamma / 2) * (s0 / (1 + s0 + 4 * (Delta / Gamma)**2))
    return R_sc



    

def scatterplot(state:str, lam_in, lam_en):
    state_list = {"1S0": 1, "3P0": 2}
    lambda_vals = np.arange(lam_in, lam_en, 0.01)

    c = 299792458
    epsilon0 = 8.8541878128e-12
    h = 6.62607004e-34
    hbar = h/2/np.pi
    a0 = 0.52917721067e-10
    au = (4 * np.pi * epsilon0 * a0**3)
    auh = (4 * np.pi * epsilon0 * a0**3) / h
    e = 1.602176634e-19           # C, elementary charge
    ea0 = e * a0                        # atomic unit of dipole moment (C*m)

    sc = []

    sc = scatterrate(lambda_vals, state_list[state])
    

    # -------------------------------
    # Prepare results folder
    # -------------------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------------
    # Plot
    # -------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(lambda_vals, sc, label=f"{state}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("R_sc/I m^2/(W s)")
    plt.xlim(lam_in, lam_en)
    #plt.ylim(-40, 40)
    plt.title("scattering rate vs Wavelength")
    plt.grid(True)
    plt.legend()

    # Save figure
    filename_png = os.path.join(results_dir, f"scatter_rate_{state}.png")
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figure saved as: {filename_png}")
    




def intensity(power_mW, waist_um):
    power_W = power_mW * 1e-3
    waist_m = waist_um * 1e-6
    I_0 = 2 * power_W / (np.pi * waist_m**2)
    return I_0

I = intensity(0.02, 100)
rate = single_transition_scattering(2*np.pi*80*1e6, I, 2*np.pi*180*1e3)
print(rate)



    