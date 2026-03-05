import numpy as np
from sympy.physics.wigner import wigner_6j
from sympy import Rational
import math

def polarizability(lambda_nm, istate, mi = 0, p = 0, I = 0, beta = 0):
    """
    polarizability of yb174(I=0 boson):

    # lambda=350:1:1100; % evaluation wavelengths in nm
    # istate=1; % evaluation state 1S0=1, 3P0=2, 3P1=3, 3P2=4
    # mi; % evaluation mF {-F:F}
    # p; % evaluation polarization {-1,0,1}
    # I; % nuclear spin for Yb isotope (0 for 174Yb, 1/2 for 171Yb and 5/2 for 173Yb)
    # beta (deg); angle between polarization and quantization axis (only defined for linear polarization light)
            for circular polarization, the propagation direction is always choosed to be parallel to 
            quantization axis
    """
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
    Fi = {}
    
    # (6s2)1S0: Thesis: "Interorbital spin exchange in a state-dependent optical
    # lattice" table B.1 + paper"Determination of static dipole polarizabilities of Yb atom"
    couplestate[1] = ['(6s6p)3P1', '(6s6p)1P1', '(7/2,5/2)o', '(6s7p)1P1', '(6s7p)3P1', '(6s8p)1P1', '(6s8p)3P1']
    wavenumber[1] = np.array([17992.00699, 25068.222, 28857.014, 40563.97, 38174.17, 44017.6, 43659.38])
    w[1] = 2 * np.pi * c * wavenumber[1] * 100  # angular frequency
    Jk[1] = np.ones_like(w[1], dtype=int)
    Ji[1] = 0
    Fi[1] = Ji[1] + I
    RME[1] = np.array([0.543, 4.148, 2.04, 0.67, 0.08, 0.21, 0.014])
    RME_SI[1] = RME[1] * ea0
    #ME[1] = (3*h*c**3*(2*Ji[1]+1)*Aj[1]/(2*np.pi*w[1]**3))**2
    
    # (6s6p)3P0: Thesis: "Interorbital spin exchange in a state-dependent optical
    # lattice" table B.1 + paper"Determination of static dipole polarizabilities of Yb atom"
    couplestate[2] = ['(5d6s)3D1', '(6s6d)3D1', '(6S7d)3D1', '(6s7s)3S1', '(6s8s)3S1', '(6p2)3P1', 'eff']
    wavenumber[2] = np.array([24489.102, 39808.72, 44311.38, 32694.692, 41615.04, 43805.42, 1 / (375 * 1e-7) + 17288.439 ]) - 17288.439
    w[2] = 2 * np.pi * c * wavenumber[2] * 100  # angular frequency
    Jk[2] = np.ones_like(w[2], dtype=int)
    Ji[2] = 0
    Fi[2] = Ji[2] + I
    RME[2] = np.array([2.96, 1.79, 2.10, 1.95, 0.62, 1.97, 2.25])
    RME_SI[2] = RME[2] * ea0
    #ME[2] = (3*h*c**3*(2*Ji[1]+1)*Aj[1]/(2*np.pi*w[1]**3))**2

    # (6s6p)3P1: Thesis: "Interorbital spin exchange in a state-dependent optical
    # lattice" table B.1 + paper"Determination of static dipole polarizabilities of Yb atom"
    couplestate[3] = ['(6s2)1S0', '(5d6s)3D1', '(5d6S)3D2', '(5d6s)1D2', '(6s6d)3D1', '(6s6d)3D2', '(6s6d)1D2', '(6s7d)3D1', '(6s7d)3D2', '(6s7d)1D2', '(6s7s)3S1', '(6s7s)1S0', '(6s8s)3S1', '(6s8s)1S0', '(6p2)3P0', '(6p2)3P1', '(6p2)3P2']
    wavenumber[3] = np.array([0, 24489.102, 24751.948, 27677.665, 39808.72, 39838.04, 40061.51, 44311.38, 44313.05, 44357.60, 32694.692, 34350.65, 41615.04, 41939.90, 42436.91, 43805.42, 44760.37]) - 17992.007
    w[3] = 2 * np.pi * c * wavenumber[3] * 100  # angular frequency
    Jk[3] = np.array([0,1,2,2,1,2,2,1,2,2,1,0,1,0,0,1,2])
    Ji[3] = 1
    Fi[3] = Ji[3] + I
    RME[3] = np.array([0.543, 2.57, 4.45, 0.46, 1.57, 2.72, 0.53, 2.47, 1.67, 0.86, 3.47, 0.24, 1.00, 0.29, 2.59, 0.18, 2.92])
    RME_SI[3] = RME[3] * ea0
    #ME[3] = (3*h*c**3*(2*Ji[1]+1)*Aj[1]/(2*np.pi*w[1]**3))**2

    # (6s6p)3P2: Thesis: "Interorbital spin exchange in a state-dependent optical
    # lattice" table B.1 + paper"Determination of static dipole polarizabilities of Yb atom"
    couplestate[4] = ['(5d6s)3D1', '(5d6s)3D2', '(5d6s)3D3', '(5d6s)1D2', '(6s7s)3S1']
    wavenumber[4] = np.array([24489.102, 24751.948, 25270.902, 27677.665, 32694.692]) - 19710.388
    w[4] = 2 * np.pi * c * wavenumber[4] * 100  # angular frequency
    Jk[4] = np.array([1,2,3,2,1])
    Ji[4] = 2
    Fi[4] = Ji[4] + I
    RME[4] = np.array([0.60, 2.39, 6.12, 0.38, 4.99])
    RME_SI[4] = RME[4] * ea0
    #ME[4] = (3*h*c**3*(2*Ji[1]+1)*Aj[1]/(2*np.pi*w[1]**3))**2

    # (5d6s)3D1: Thesis: "Interorbital spin exchange in a state-dependent optical
    # lattice" table B.1 + paper"Determination of static dipole polarizabilities of Yb atom"
    couplestate[5] = ['(6s6p)3P0', '(6s6p)3P1', '(6s6p)3P2', '(6s6p)1P1']
    wavenumber[5] = np.array([17288.439, 17992.007, 19710.388, 25068.222]) - 24489.102
    w[5] = 2 * np.pi * c * wavenumber[5] * 100  # angular frequency
    Jk[5] = np.array([0,1,2,1])
    Ji[5] = 1
    Fi[5] = Ji[5] + I
    RME[5] = np.array([2.96, 2.57, 0.60, 0.27])
    RME_SI[5] = RME[5] * ea0
    #ME[4] = (3*h*c**3*(2*Ji[1]+1)*Aj[1]/(2*np.pi*w[1]**3))**2

    # (5d6s)3D2: Thesis: "Interorbital spin exchange in a state-dependent optical
    # lattice" table B.1 + paper"Determination of static dipole polarizabilities of Yb atom"
    couplestate[6] = ['(6s6p)3P1', '(6s6p)3P2', '(6s6p)1P1']
    wavenumber[6] = np.array([17992.007, 19710.388, 25068.222]) - 24751.948
    w[6] = 2 * np.pi * c * wavenumber[6] * 100  # angular frequency
    Jk[6] = np.array([1,2,1])
    Ji[6] = 2
    Fi[6] = Ji[6] + I
    RME[6] = np.array([4.45, 2.39, 0.32])
    RME_SI[6] = RME[6] * ea0
    #ME[4] = (3*h*c**3*(2*Ji[1]+1)*Aj[1]/(2*np.pi*w[1]**3))**2

    # (6s7s)3S1: Thesis: "Interorbital spin exchange in a state-dependent optical
    # lattice" table B.1 + paper"Determination of static dipole polarizabilities of Yb atom"
    couplestate[7] = ['(6s6p)3P0', '(6s6p)3P1', '(6s6p)3P2', '(6s6p)1P1']
    wavenumber[7] = np.array([17288.439, 17992.007, 19710.388, 25068.222]) - 32694.692
    w[7] = 2 * np.pi * c * wavenumber[7] * 100  # angular frequency
    Jk[7] = np.array([0,1,2,1])
    Ji[7] = 1
    Fi[7] = Ji[7] + I
    RME[7] = np.array([1.95, 3.47, 5.05, 0.73])
    RME_SI[7] = RME[7] * ea0
    #ME[4] = (3*h*c**3*(2*Ji[1]+1)*Aj[1]/(2*np.pi*w[1]**3))**2




    wlight = 2 * np.pi * c / (np.array(lambda_nm) * 1e-9)
    alpha_s = np.zeros_like(wlight, dtype=float)
    alpha_v = np.zeros_like(wlight, dtype=float)
    alpha_t = np.zeros_like(wlight, dtype=float)




    if istate == 8:
        # not used in current tables above, but kept for completeness
        for k in range(len(Jk[istate])):
            alpha += (2 / 3 / (2 * Jk[istate][k] + 1)
                      * w[istate][k] / (w[istate][k]**2 - wlight**2) * RME_SI[istate][k]**2)
    else:
        if Fi[istate] == 0:
            if p == 0:
                for k in range(len(Jk[istate])):
                    alpha_s += (2/3/hbar) * (w[istate][k]/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 / (2*Ji[istate]+1)
                alpha = alpha_s
                return -alpha/au
            elif p == 1:
                for k in range(len(Jk[istate])):
                    alpha_s += (2/3/hbar) * (w[istate][k]/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 / (2*Ji[istate]+1)
                alpha = alpha_s
                return -alpha/au
            elif p == -1:
                for k in range(len(Jk[istate])):
                    alpha_s += (2/3/hbar) * (w[istate][k]/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 / (2*Ji[istate]+1)
                alpha = alpha_s
                return -alpha/au
            else:
                raise ValueError("Invalid polarization")
        elif Fi[istate] == 0.5:
            if p == 0:
                for k in range(len(Jk[istate])):
                    alpha_s += (2/3/hbar) * (w[istate][k]/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 / (2*Ji[istate]+1)
                alpha = alpha_s
                return -alpha/au
            elif p == 1:
                for k in range(len(Jk[istate])):
                    alpha_s += (2/3/hbar) * (w[istate][k]/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 / (2*Ji[istate]+1)
                    alpha_v += (-1)**(2*Ji[istate]+Jk[istate][k]+Fi[istate]+I) * (6*Fi[istate]*(2*Fi[istate]+1)/(Fi[istate]+1))**0.5 * float(wigner_6j(1,1,1,Ji[istate],Ji[istate],Jk[istate][k])) * float(wigner_6j(Ji[istate],Ji[istate],1,Fi[istate],Fi[istate],I)) * mi/Fi[istate] * (wlight/hbar/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 
                alpha = alpha_s + alpha_v
                return -alpha/au
            elif p == -1:
                for k in range(len(Jk[istate])):
                    alpha_s += (2/3/hbar) * (w[istate][k]/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 / (2*Ji[istate]+1)
                    alpha_v += (-1)**(2*Ji[istate]+Jk[istate][k]+Fi[istate]+I) * (6*Fi[istate]*(2*Fi[istate]+1)/(Fi[istate]+1))**0.5 * float(wigner_6j(1,1,1,Ji[istate],Ji[istate],Jk[istate][k])) * float(wigner_6j(Ji[istate],Ji[istate],1,Fi[istate],Fi[istate],I)) * mi/Fi[istate] * (wlight/hbar/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2
                alpha = alpha_s - alpha_v
                return -alpha/au
            else:
                raise ValueError("Invalid polarization")
        else:
            if p == 0:
                for k in range(len(Jk[istate])):
                    alpha_s += (2/3/hbar) * (w[istate][k]/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 / (2*Ji[istate]+1)
                    alpha_t += (-1)**(2*Ji[istate]+Jk[istate][k]+Fi[istate]+I) * (40*Fi[istate]*(2*Fi[istate]+1)*(2*Fi[istate]-1)/3/(Fi[istate]+1)/(2*Fi[istate]+3))**0.5 * float(wigner_6j(1,1,2,Ji[istate],Ji[istate],Jk[istate][k])) * float(wigner_6j(Ji[istate],Ji[istate],2,Fi[istate],Fi[istate],I)) * (w[istate][k]/hbar/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 * (3*mi**2-Fi[istate]*(Fi[istate]+1))/Fi[istate]/(2*Fi[istate]-1)
                alpha = alpha_s + (3*math.cos(beta*np.pi/180)**2-1)*alpha_t/2
                return -alpha/au
            elif p == 1:
                for k in range(len(Jk[istate])):
                    alpha_s += (2/3/hbar) * (w[istate][k]/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 / (2*Ji[istate]+1)
                    alpha_v += (-1)**(2*Ji[istate]+Jk[istate][k]+Fi[istate]+I) * (6*Fi[istate]*(2*Fi[istate]+1)/(Fi[istate]+1))**0.5 * float(wigner_6j(1,1,1,Ji[istate],Ji[istate],Jk[istate][k])) * float(wigner_6j(Ji[istate],Ji[istate],1,Fi[istate],Fi[istate],I)) * mi/Fi[istate] * (wlight/hbar/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 
                    alpha_t += (-1)**(2*Ji[istate]+Jk[istate][k]+Fi[istate]+I) * (40*Fi[istate]*(2*Fi[istate]+1)*(2*Fi[istate]-1)/3/(Fi[istate]+1)/(2*Fi[istate]+3))**0.5 * float(wigner_6j(1,1,2,Ji[istate],Ji[istate],Jk[istate][k])) * float(wigner_6j(Ji[istate],Ji[istate],2,Fi[istate],Fi[istate],I)) * (w[istate][k]/hbar/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 * (3*mi**2-Fi[istate]*(Fi[istate]+1))/Fi[istate]/(2*Fi[istate]-1)
                alpha = alpha_s + alpha_v - 0.5 * alpha_t
                return -alpha/au
            elif p == -1:
                for k in range(len(Jk[istate])):
                    alpha_s += (2/3/hbar) * (w[istate][k]/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 / (2*Ji[istate]+1)
                    alpha_v += (-1)**(2*Ji[istate]+Jk[istate][k]+Fi[istate]+I) * (6*Fi[istate]*(2*Fi[istate]+1)/(Fi[istate]+1))**0.5 * float(wigner_6j(1,1,1,Ji[istate],Ji[istate],Jk[istate][k])) * float(wigner_6j(Ji[istate],Ji[istate],1,Fi[istate],Fi[istate],I)) * mi/Fi[istate] * (wlight/hbar/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2
                    alpha_t += (-1)**(2*Ji[istate]+Jk[istate][k]+Fi[istate]+I) * (40*Fi[istate]*(2*Fi[istate]+1)*(2*Fi[istate]-1)/3/(Fi[istate]+1)/(2*Fi[istate]+3))**0.5 * float(wigner_6j(1,1,2,Ji[istate],Ji[istate],Jk[istate][k])) * float(wigner_6j(Ji[istate],Ji[istate],2,Fi[istate],Fi[istate],I)) * (w[istate][k]/hbar/(w[istate][k]**2 - wlight**2)) * RME_SI[istate][k]**2 * (3*mi**2-Fi[istate]*(Fi[istate]+1))/Fi[istate]/(2*Fi[istate]-1)
                alpha = alpha_s - alpha_v - 0.5 * alpha_t
                return -alpha/au
            else:
                raise ValueError("Invalid polarization")

print(polarizability(556.01, 1))
print(polarizability(532, 1))