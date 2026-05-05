import numpy as np
from scipy.constants import c, h, epsilon_0, e, pi
hbar = h / (2 * pi)
a0 = 0.52917721067e-10
ea0 = e * a0  # RME 的原子單位

def calc_RME_au(Gamma, lambda_nm, J_upper):
    """將 Transition Rate (Hz) 轉換為 RME (a.u.)"""
    if Gamma == 0:
        return 0.0
    
    w = 2 * pi * c / (lambda_nm * 1e-9)
    
    # 反推 RME 的 SI 單位平方
    RME_SI_sq = ((Gamma * 2 * pi) * 3 * pi * epsilon_0 * hbar * c**3 * (2 * J_upper + 1)) / (w**3)
    
    # 開根號並轉換為原子單位 (a.u.)
    RME_au = np.sqrt(RME_SI_sq) / ea0
    return RME_au

def calc_Gamma_Hz(RME, lambda_nm, J_upper):
    if RME == 0:
        return 0.0
    
    w = 2 * pi * c / (lambda_nm * 1e-9)
    Gamma = (w**3) / ((RME * ea0)**2 * 3 * pi * epsilon_0 * hbar * c**3 * (2 * J_upper + 1))
    return Gamma / 2 / pi

print(calc_RME_au(29*1e6,398.9,1))
