import streamlit as st
import numpy as np
import plotly.graph_objects as go
from polarizabilityyb1 import polarizability

st.set_page_config(page_title="Yb Polarizability Calculator", layout="wide")

# --- 側邊欄：全域參數設定 ---
st.sidebar.title("全域參數設定")

isotope = st.sidebar.selectbox("同位素 (Isotope)", [174, 171, 173])
I_dict = {174: 0.0, 171: 0.5, 173: 2.5}
I_val = I_dict[isotope]

unit_choice = st.sidebar.radio("縱軸單位 (Y-axis Unit)", 
                               ["Atomic Unit (a.u.)", "Stark Shift (h Hz W^-1 cm^2)"])

pol_str = st.sidebar.selectbox("光偏振 (Polarization)", 
                               ["pi (Linear)", "sigma+ (Circular)", "sigma- (Circular)"])
pol_dict = {"pi (Linear)": 0, "sigma+ (Circular)": 1, "sigma- (Circular)": -1}
p_val = pol_dict[pol_str]

if p_val == 0:
    beta_angle = st.sidebar.number_input("入射角度 (Beta in deg)", min_value=0.0, max_value=180.0, value=0.0)
else:
    beta_angle = 0.0

# --- 主畫面：能階與 mF 選擇 ---
st.title("Ytterbium (Yb) 互動式極化率計算機")
st.markdown("請勾選能階，並選擇 $m_F$ 值。圖表支援**滑鼠懸停數值、滾輪縮放、拖拉平移**。")

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

cols = st.columns(4)
for i, (state_name, info) in enumerate(states_info.items()):
    with cols[i % 4]:
        st.markdown(f"**{state_name}**")
        is_checked = st.checkbox(f"顯示 {state_name}", key=f"chk_{state_name}")
        
        if is_checked:
            F_val = info["J"] + I_val
            # 恢復正負號選單：從 -F 到 F
            mF_options = np.arange(-F_val, F_val + 1, 1.0)
            selected_mF = st.selectbox(f"選擇 mF", mF_options, key=f"mf_{state_name}")
            
            selected_configs.append({
                "name": state_name,
                "istate": info["istate"],
                "J": info["J"],
                "mF": selected_mF 
            })

st.divider()

# --- 繪圖邏輯 (使用 Plotly) ---
if len(selected_configs) > 0:
    # 增加波長解析度，讓共振峰更精細
    wavelengths = np.linspace(300, 800, 4000) 
    
    c = 299792458
    h = 6.62607004e-34
    a0 = 0.52917721067e-10
    
    fig = go.Figure()
    
    for config in selected_configs:
        pol_val_array = polarizability(wavelengths, config["istate"], config["mF"], p_val, I_val, beta_angle)
        
        if unit_choice == "Atomic Unit (a.u.)":
            y_plot = -pol_val_array
            y_limit = 1000
            ylabel_str = "Polarizability (a.u.)"
        else:
            factor = (2 * np.pi * a0**3 / (c * h)) * 10000 
            y_plot = pol_val_array * factor
            y_limit = 40
            ylabel_str = "V_ac / I (h Hz W^-1 cm^2)"

        y_plot = np.array(y_plot, dtype=float)
        
        # 【全新斷線演算法】：利用微積分概念，偵測相鄰兩點差異過大的地方 (跨越共振漸近線)
        diffs = np.abs(np.diff(y_plot))
        jump_threshold = y_limit * 2  # 當兩點之間跳躍超過視野兩倍時斷開
        jump_indices = np.where(diffs > jump_threshold)[0]
        
        for idx in jump_indices:
            y_plot[idx] = np.nan
            y_plot[idx+1] = np.nan
            
        label_str = f"{config['name']}, F={config['J']+I_val}, mF={config['mF']}"
        
        # 加入 Plotly 曲線
        fig.add_trace(go.Scatter(
            x=wavelengths, 
            y=y_plot, 
            mode='lines', 
            name=label_str,
            hovertemplate="波長: %{x:.2f} nm<br>極化率: %{y:.2f}<extra></extra>"
        ))

    # 設定 Plotly 圖表外觀與互動功能
    fig.update_layout(
        xaxis_title="Wavelength (nm)",
        yaxis_title=ylabel_str,
        yaxis_range=[-y_limit, y_limit], # 強制鎖定預設的 Y 軸顯示範圍，但保留 Peak 資料
        xaxis_range=[300, 800],
        hovermode="x unified", # 游標移上去會顯示同一垂直線上的所有數值
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("請在上方至少勾選一個能階來顯示圖表。")