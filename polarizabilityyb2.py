import numpy as np
from sympy.physics.wigner import wigner_6j
import math

# 載入我們寫好的 JSON 資料庫引擎
from atomic_data_base import YbAtomicDatabase

# ==========================================
# 1. 物理常數初始化
# ==========================================
c = 299792458
h = 6.62607004e-34
hbar = h / (2 * np.pi)
epsilon0 = 8.8541878128e-12
a0 = 0.52917721067e-10
au = (4 * np.pi * epsilon0 * a0**3)
e = 1.602176634e-19           
ea0 = e * a0                  

# 初始化資料庫 (確保 atomdata.json 在同一目錄下)
db = YbAtomicDatabase('atomdata.json')

# ==========================================
# 2. 核心極化率函數
# ==========================================
def polarizability(lambda_nm, state, mi=0, p=0, I=0, beta=0):
    """
    極化率計算機:
    :param lambda_nm: 波長陣列或單一波長 (nm)
    :param state: 能階字串 (如 "1S0", "3P1") 或舊版整數代號 (1~7)
    :param mi: 磁量子數 mF
    :param p: 偏振 (0: linear, 1: sigma+, -1: sigma-)
    :param I: 核自旋 (174Yb 為 0, 171Yb 為 0.5, 173Yb 為 2.5)
    :param beta: 光傳播方向或極化向量與量子化軸的夾角 (degree)
    """
    
    # ---------------------------------------------------------
    # A. 參數前處理與資料庫查詢
    # ---------------------------------------------------------
    # 防呆/兼容機制：如果舊版網頁傳入數字，自動轉換為 JSON 能階名稱
    if isinstance(state, int):
        legacy_map = {1: "1S0", 2: "3P0", 3: "3P1", 4: "3P2", 5: "3D1", 6: "3D2", 7: "3S1_6s7s"}
        state_name = legacy_map.get(state, "1S0")
    else:
        state_name = state

    # 呼叫資料庫，直接拿取該能階對應的所有耦合參數
    couplings = db.get_couplings(state_name)
    
    # 將變數攤平，徹底拔除舊版 [istate] 的維度包袱
    Ji = couplings['J_i']
    w = couplings['w']
    Jk = couplings['J_k']
    RME_SI = couplings['RME_au'] * ea0  # 從 a.u. 轉回 SI 單位
    Fi = Ji + I

    # ---------------------------------------------------------
    # B. 極化率公式計算 (完全保留原有張量推導，僅將陣列降維)
    # ---------------------------------------------------------
    wlight = 2 * np.pi * c / (np.array(lambda_nm) * 1e-9)
    alpha_s = np.zeros_like(wlight, dtype=float)
    alpha_v = np.zeros_like(wlight, dtype=float)
    alpha_t = np.zeros_like(wlight, dtype=float)

    beta_rad = beta * np.pi / 180

    if Fi == 0:
        if p == 0 or p == 1 or p == -1:
            for k in range(len(Jk)):
                alpha_s += (2/3/hbar) * (w[k]/(w[k]**2 - wlight**2)) * RME_SI[k]**2 / (2*Ji+1)
            alpha = alpha_s
            return -alpha/au
        else:
            raise ValueError("Invalid polarization")
            
    elif Fi == 0.5:
        if p == 0:
            for k in range(len(Jk)):
                alpha_s += (2/3/hbar) * (w[k]/(w[k]**2 - wlight**2)) * RME_SI[k]**2 / (2*Ji+1)
            alpha = alpha_s
            return -alpha/au
        elif p == 1:
            for k in range(len(Jk)):
                alpha_s += (2/3/hbar) * (w[k]/(w[k]**2 - wlight**2)) * RME_SI[k]**2 / (2*Ji+1)
                alpha_v += (-1)**(2*Ji+Jk[k]+Fi+I) * (6*Fi*(2*Fi+1)/(Fi+1))**0.5 * float(wigner_6j(1,1,1,Ji,Ji,Jk[k])) * float(wigner_6j(Ji,Ji,1,Fi,Fi,I)) * mi/Fi * (wlight/hbar/(w[k]**2 - wlight**2)) * RME_SI[k]**2 
            alpha = alpha_s + alpha_v * math.cos(beta_rad)
            return -alpha/au
        elif p == -1:
            for k in range(len(Jk)):
                alpha_s += (2/3/hbar) * (w[k]/(w[k]**2 - wlight**2)) * RME_SI[k]**2 / (2*Ji+1)
                alpha_v += (-1)**(2*Ji+Jk[k]+Fi+I) * (6*Fi*(2*Fi+1)/(Fi+1))**0.5 * float(wigner_6j(1,1,1,Ji,Ji,Jk[k])) * float(wigner_6j(Ji,Ji,1,Fi,Fi,I)) * mi/Fi * (wlight/hbar/(w[k]**2 - wlight**2)) * RME_SI[k]**2
            alpha = alpha_s - alpha_v * math.cos(beta_rad)
            return -alpha/au
        else:
            raise ValueError("Invalid polarization")
            
    else:  # Fi > 0.5
        if p == 0:
            for k in range(len(Jk)):
                alpha_s += (2/3/hbar) * (w[k]/(w[k]**2 - wlight**2)) * RME_SI[k]**2 / (2*Ji+1)
                alpha_t += (-1)**(2*Ji+Jk[k]+Fi+I) * (40*Fi*(2*Fi+1)*(2*Fi-1)/3/(Fi+1)/(2*Fi+3))**0.5 * float(wigner_6j(1,1,2,Ji,Ji,Jk[k])) * float(wigner_6j(Ji,Ji,2,Fi,Fi,I)) * (w[k]/hbar/(w[k]**2 - wlight**2)) * RME_SI[k]**2 * (3*mi**2-Fi*(Fi+1))/Fi/(2*Fi-1)
            alpha = alpha_s + (3*math.cos(beta_rad)**2-1)*alpha_t/2
            return -alpha/au
        elif p == 1:
            for k in range(len(Jk)):
                alpha_s += (2/3/hbar) * (w[k]/(w[k]**2 - wlight**2)) * RME_SI[k]**2 / (2*Ji+1)
                alpha_v += (-1)**(2*Ji+Jk[k]+Fi+I) * (6*Fi*(2*Fi+1)/(Fi+1))**0.5 * float(wigner_6j(1,1,1,Ji,Ji,Jk[k])) * float(wigner_6j(Ji,Ji,1,Fi,Fi,I)) * mi/Fi * (wlight/hbar/(w[k]**2 - wlight**2)) * RME_SI[k]**2 
                alpha_t += (-1)**(2*Ji+Jk[k]+Fi+I) * (40*Fi*(2*Fi+1)*(2*Fi-1)/3/(Fi+1)/(2*Fi+3))**0.5 * float(wigner_6j(1,1,2,Ji,Ji,Jk[k])) * float(wigner_6j(Ji,Ji,2,Fi,Fi,I)) * (w[k]/hbar/(w[k]**2 - wlight**2)) * RME_SI[k]**2 * (3*mi**2-Fi*(Fi+1))/Fi/(2*Fi-1)
            alpha = alpha_s + alpha_v * math.cos(beta_rad) + (3*(math.sin(beta_rad)**2)/2 - 1)/2 * alpha_t
            return -alpha/au
        elif p == -1:
            for k in range(len(Jk)):
                alpha_s += (2/3/hbar) * (w[k]/(w[k]**2 - wlight**2)) * RME_SI[k]**2 / (2*Ji+1)
                alpha_v += (-1)**(2*Ji+Jk[k]+Fi+I) * (6*Fi*(2*Fi+1)/(Fi+1))**0.5 * float(wigner_6j(1,1,1,Ji,Ji,Jk[k])) * float(wigner_6j(Ji,Ji,1,Fi,Fi,I)) * mi/Fi * (wlight/hbar/(w[k]**2 - wlight**2)) * RME_SI[k]**2
                alpha_t += (-1)**(2*Ji+Jk[k]+Fi+I) * (40*Fi*(2*Fi+1)*(2*Fi-1)/3/(Fi+1)/(2*Fi+3))**0.5 * float(wigner_6j(1,1,2,Ji,Ji,Jk[k])) * float(wigner_6j(Ji,Ji,2,Fi,Fi,I)) * (w[k]/hbar/(w[k]**2 - wlight**2)) * RME_SI[k]**2 * (3*mi**2-Fi*(Fi+1))/Fi/(2*Fi-1)
            alpha = alpha_s - alpha_v * math.cos(beta_rad) + (3*(math.sin(beta_rad)**2)/2 - 1)/2 * alpha_t
            return -alpha/au
        else:
            raise ValueError("Invalid polarization")