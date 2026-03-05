import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from polarizabilityyb1 import polarizability

# --- 網頁全域設定 ---
st.set_page_config(page_title="Yb Polarizability Calculator", layout="wide")

# --- 側邊欄：參數設定 ---
st.sidebar.title("全域參數設定")

# 選擇同位素，並自動對應核自旋 I
isotope = st.sidebar.selectbox("同位素 (Isotope)", [174, 171, 173])
I_dict = {174: 0.0, 171: 0.5, 173: 2.5}
I_val = I_dict[isotope]

# 選擇圖表縱軸單位
unit_choice = st.sidebar.radio("縱軸單位 (Y-axis Unit)", 
                               ["Atomic Unit (a.u.)", "Stark Shift (h Hz W^-1 cm^2)"])

# 選擇光偏振狀態
pol_str = st.sidebar.selectbox("光偏振 (Polarization)", 
                               ["pi (Linear)", "sigma+ (Circular)", "sigma- (Circular)"])
pol_dict = {"pi (Linear)": 0, "sigma+ (Circular)": 1, "sigma- (Circular)": -1}
p_val = pol_dict[pol_str]

# 如果是線性偏振，才顯示入射角度輸入框
beta_angle = 0.0
if p_val == 0:
    beta_angle = st.sidebar.number_input("入射角度 (Beta in deg)", min_value=0.0, max_value=180.0, value=0.0)


# --- 主畫面：能階與 mF 選擇 ---
st.title("Ytterbium (Yb) 互動式極化率計算機")
st.markdown("請在下方勾選您要繪製的能階，程式會根據您選擇的同位素自動計算 $F$ 值，並提供對應的 $m_F$ 選單。")

# 建立 7 個能階的字典 (對應 polarizabilityyb1.py 中的 istate 與理論 J 值)
states_info = {
    "1S0": {"istate": 1, "J": 0},
    "3P0": {"istate": 2, "J": 0},
    "3P1": {"istate": 3, "J": 1},
    "3P2": {"istate": 4, "J": 2},
    "(5d6s)3D1": {"istate": 5, "J": 1},
    "(5d6s)3D2": {"istate": 6, "J": 2},
    "(6s7s)3S1": {"istate": 7, "J": 1}
}

selected_configs = []

# 使用 4 個欄位來整齊排版 Checkbox 與 Selectbox
cols = st.columns(4)
for i, (state_name, info) in enumerate(states_info.items()):
    with cols[i % 4]:
        st.markdown(f"**{state_name}**")
        is_checked = st.checkbox(f"顯示 {state_name}", key=f"chk_{state_name}")
        
        if is_checked:
            # 計算 F = J + I
            F_val = info["J"] + I_val
            # 產生從 -F 到 F 的陣列 (步長為 1)
            mF_options = np.arange(-F_val, F_val + 1, 1.0)
            
            # 動態顯示該能階的 mF 下拉選單
            selected_mF = st.selectbox(f"選擇 mF", mF_options, key=f"mf_{state_name}")
            
            selected_configs.append({
                "name": state_name,
                "istate": info["istate"],
                "J": info["J"],
                "mF": selected_mF
            })

st.divider()

# --- 繪圖邏輯 ---
if len(selected_configs) > 0:
    fig, ax = plt.subplots(figsize=(12, 7))
    wavelengths = np.linspace(300, 800, 1500) # 設定波長範圍與解析度
    
    # 物理常數 (用於單位轉換)
    c = 299792458
    h = 6.62607004e-34
    a0 = 0.52917721067e-10
    
    for config in selected_configs:
        # 呼叫你寫好的物理公式模組
        # 回傳值預設為 -alpha_au (紅失諧時為負值)
        pol_val_array = polarizability(wavelengths, config["istate"], config["mF"], p_val, I_val, beta_angle)
        
        if unit_choice == "Atomic Unit (a.u.)":
            # 為了符合常規 a.u. 的習慣（靜態極化率為正），將原始輸出乘上 -1
            y_plot = -pol_val_array
            y_limit = 1000
            ylabel_str = "Polarizability (a.u.)"
        else:
            # Stark Shift 轉換係數：從 -alpha_au 轉換為 h Hz W^-1 cm^2
            # 轉換因子 = (2 * pi * a0^3 / (c * h)) * 10000
            factor = (2 * np.pi * a0**3 / (c * h)) * 10000 
            y_plot = pol_val_array * factor
            y_limit = 40
            ylabel_str = "$V_{ac} / I$ ($h$ Hz W$^{-1}$ cm$^2$)"

        y_plot = np.array(y_plot, dtype=float)
        
        # 去除共振波長的垂直連線 (漸近線處理)
        threshold = y_limit * 1.5
        y_plot[np.abs(y_plot) > threshold] = np.nan
        
        # 設定圖例標籤
        label_str = f"{config['name']}, F={config['J']+I_val}, mF={config['mF']}"
        ax.plot(wavelengths, y_plot, label=label_str)

    # 圖表美化與設定
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlim(300, 800)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel(ylabel_str, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc="upper right")
    
    # 在網頁上顯示 Matplotlib 圖表
    st.pyplot(fig)
else:
    st.info("請在上方至少勾選一個能階來顯示圖表。")