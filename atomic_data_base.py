import json
import numpy as np
from calc_RME_au import calc_RME_au
from scipy.constants import c, h, epsilon_0, e, pi
hbar = h / (2 * pi)
a0 = 0.52917721067e-10
ea0 = e * a0  # RME 的原子單位

class YbAtomicDatabase:
    def __init__(self, json_filepath):
        # 1. 讀取 JSON
        with open(json_filepath, 'r') as f:
            data = json.load(f)
            
        # 2. 建立能階字典 (State Dictionary)
        # 格式: {'1S0': {'J': 0, 'energy': 0.0}, '3P1': {'J': 1, 'energy': 17992.007}, ...}
        self.states = {}
        for state in data['states']:
            name, J, energy = state
            self.states[name] = {'J': J, 'energy': energy}
            
        # 3. 建立躍遷資料庫並自動計算 RME
        self.transitions = []
        for trans in data['transitions']:
            upper, lower, wvl, A_ji = trans
            J_upper = self.states[upper]['J']
            
            # 自動呼叫轉換函數
            rme_au = calc_RME_au(A_ji, wvl, J_upper)
            
            self.transitions.append({
                'upper': upper,
                'lower': lower,
                'wavelength_nm': wvl,
                'A_rate': A_ji,
                'RME_au': rme_au
            })

    def get_couplings(self, target_state_name):
        """
        給定一個能階 (例如 '1S0')，自動找尋所有與它耦合的躍遷，
        並回傳極化率公式所需要的矩陣陣列。
        """
        w_list = []
        Jk_list = []
        RME_list = []
        
        target_energy = self.states[target_state_name]['energy']
        
        # 尋找所有包含此能階的躍遷 (包含往上與往下)
        for t in self.transitions:
            if t['lower'] == target_state_name or t['upper'] == target_state_name:
                
                # 判斷耦合態是哪個
                coupled_state = t['upper'] if t['lower'] == target_state_name else t['lower']
                coupled_energy = self.states[coupled_state]['energy']
                
                # 躍遷頻率 (取決於兩能階能量差，確保正負號正確)
                # 這裡使用你原本邏輯: w = 2*pi*c * (wavenumber差) * 100
                delta_wavenumber = abs(coupled_energy - target_energy)
                w_angular = 2 * pi * 299792458 * delta_wavenumber * 100
                
                w_list.append(w_angular)
                Jk_list.append(self.states[coupled_state]['J'])
                RME_list.append(t['RME_au'])
                
        return {
            'J_i': self.states[target_state_name]['J'],
            'w': np.array(w_list),
            'J_k': np.array(Jk_list, dtype=int),
            'RME_au': np.array(RME_list)
        }