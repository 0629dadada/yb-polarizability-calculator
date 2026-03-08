import streamlit as st
import numpy as np
import plotly.graph_objects as go
from polarizabilityyb1 import polarizability
from scattering_rate import scatterrate

st.set_page_config(page_title="Yb Physics Calculator", layout="wide")

# --- 側邊欄：全域參數設定 ---
st.sidebar.title("全域參數設定")

st.sidebar.markdown("### 1. 物理參數 (共用)")
isotope = st.sidebar.selectbox("同位素 (Isotope)", [174, 171, 173])
I_dict = {174: 0.0, 171: 0.5, 173: 2.5}
I_val = I_dict[isotope]

pol_str = st.sidebar.selectbox("光偏振 (Polarization)", 
                               ["pi (Linear)", "sigma+ (Circular)", "sigma- (Circular)"])
pol_dict = {"pi (Linear)": 0, "sigma+ (Circular)": 1, "sigma- (Circular)": -1}
p_val = pol_dict[pol_str]

if p_val == 0:
    beta_angle = st.sidebar.number_input("入射角度 (Beta in deg)", min_value=0.0, max_value=180.0, value=0.0)
else:
    beta_angle = 0.0

st.sidebar.markdown("### 2. 圖表顯示範圍 (僅用於繪圖)")
col1, col2 = st.sidebar.columns(2)
with col1:
    min_wl = st.number_input("最小波長 (nm)", min_value=10.0, value=300.0, step=50.0)
with col2:
    max_wl = st.number_input("最大波長 (nm)", min_value=50.0, value=800.0, step=50.0)

unit_choice = st.sidebar.radio("縱軸單位 (Y-axis Unit)", 
                               ["Atomic Unit (a.u.)", "Stark Shift (h Hz W^-1 cm^2)"])

# --- 建立雙分頁 ---
tab1, tab2 = st.tabs(["📊 Polarizability Plotter (極化率圖表)", "🧮 Trap & Scattering Calculator (光學阱與散射計算)"])

# ==========================================
# 分頁 1：極化率圖表 (原本的功能)
# ==========================================
with tab1:
    st.title("Ytterbium (Yb) 互動式極化率圖表")
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
                mF_options = np.arange(-F_val, F_val + 1, 1.0)
                selected_mF = st.selectbox(f"選擇 mF", mF_options, key=f"mf_{state_name}")
                
                selected_configs.append({
                    "name": state_name,
                    "istate": info["istate"],
                    "J": info["J"],
                    "mF": selected_mF 
                })

    st.divider()

    if len(selected_configs) > 0:
        resolution = int(max(4000, (max_wl - min_wl) * 10)) 
        wavelengths = np.linspace(min_wl, max_wl, resolution) 
        
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
            
            diffs = np.abs(np.diff(y_plot))
            jump_threshold = y_limit * 2  
            jump_indices = np.where(diffs > jump_threshold)[0]
            
            for idx in jump_indices:
                y_plot[idx] = np.nan
                y_plot[idx+1] = np.nan
                
            label_str = f"{config['name']}, F={config['J']+I_val}, mF={config['mF']}"
            
            fig.add_trace(go.Scatter(
                x=wavelengths, 
                y=y_plot, 
                mode='lines', 
                name=label_str,
                hovertemplate="波長: %{x:.2f} nm<br>數值: %{y:.2f}<extra></extra>"
            ))

        fig.update_layout(
            xaxis=dict(title="Wavelength (nm)", range=[min_wl, max_wl], rangemode="nonnegative"),
            yaxis=dict(title=ylabel_str, range=[-y_limit, y_limit]),
            hovermode="x unified", 
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("請在上方至少勾選一個能階來顯示圖表。")

# ==========================================
# 分頁 2：光學阱與散射計算機 (新增功能)
# ==========================================
with tab2:
    st.title("光學阱深度與散射率計算 (1S0 基態)")
    
    # 物理常數
    c_const = 299792458
    h_const = 6.62607004e-34
    kB = 1.380649e-23
    epsilon_0 = 8.8541878128e-12
    a0_const = 0.52917721067e-10
    au_SI = 4 * np.pi * epsilon_0 * a0_const**3
    m_Yb = 2.8733965E-25 # Flet 提供的 Yb 平均質量
    
    # 躍遷波數對應表 (單位: cm^-1)
    transitions_wn = {
        "(6s6p)3P1 (≈ 556 nm)": 17992.00699,
        "(6s6p)1P1 (≈ 399 nm)": 25068.222,
        "(7/2,5/2)o (≈ 346 nm)": 28857.014,
        "(6s7p)3P1 (≈ 262 nm)": 38174.17,
        "(6s7p)1P1 (≈ 246 nm)": 40563.97,
        "(6s8p)3P1 (≈ 229 nm)": 43659.38,
        "(6s8p)1P1 (≈ 227 nm)": 44017.6
    }

    # --- UI 輸入區塊 ---
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.subheader("1. 光源與雷射設定")
        regime = st.radio("Detuning Regime", ["Far-Detune (遠失諧)", "Near Resonance (近共振)"])
        
        if regime == "Far-Detune (遠失諧)":
            input_wvl = st.number_input("輸入波長 (nm)", value=532.0, step=1.0, format="%.4f")
            eval_wvl_nm = input_wvl
        else:
            selected_trans = st.selectbox("選擇近共振躍遷能階", list(transitions_wn.keys()))
            detuning_mhz = st.number_input("Detuning (MHz)", value=-40.0, step=1.0)
            
            # 計算等效波長
            wn = transitions_wn[selected_trans]
            f_res = c_const * wn * 100 # Hz
            f_eval = f_res + (detuning_mhz * 1e6)
            eval_wvl_nm = (c_const / f_eval) * 1e9
            
            st.caption(f"對應評估波長: **{eval_wvl_nm:.6f} nm**")
            
    with col_input2:
        st.subheader("2. 陷阱幾何與功率")
        trap_type = st.radio("陷阱型態 (Trap Type)", ["Dipole Trap (Single Beam)", "Conveyor Belt (1D Lattice)"])
        waist_um = st.number_input("Waist 半徑 (um)", value=5.0, step=0.1)
        power_mw = st.number_input("功率 Power (mW)", value=0.02, step=0.01, format="%.3f")
        
    st.divider()

    # --- 核心計算 ---
    if st.button("🚀 開始計算 (Calculate)", use_container_width=True):
        try:
            # 1. 計算強度 Intensity
            w0_m = waist_um * 1e-6
            P_W = power_mw * 1e-3
            I_0 = 2 * P_W / (np.pi * w0_m**2)
            
            if trap_type == "Conveyor Belt (1D Lattice)":
                I_calc = 4 * I_0
                st.info("ℹ️ 1D Lattice: 計算採用波腹峰值強度 (I = 4 × I_0)")
            else:
                I_calc = I_0
                st.info("ℹ️ Dipole Trap: 計算採用焦點中心強度 (I = I_0)")

            # 2. 計算 Trap Depth (假設基態 1S0, istate=1)
            # 從 polarizabilityyb1 取出的 a.u. 極化率
            alpha_au_val = abs(polarizability(eval_wvl_nm, 1, 0, p_val, I_val, beta_angle))
            alpha_SI = alpha_au_val * au_SI
            U_joule = (alpha_SI * I_calc) / (2 * epsilon_0 * c_const)
            trap_depth_uK = (U_joule / kB) * 1e6

            # 3. 計算 Scattering Rate
            rate_per_I = scatterrate(eval_wvl_nm, 1)
            R_sc = float(rate_per_I) * I_calc

            # 4. 計算 Rayleigh Range & Recoil Energy
            z_R_mm = (np.pi * w0_m**2 / (eval_wvl_nm * 1e-9)) * 1e3
            recoil_J = h_const**2 / (2 * m_Yb * (eval_wvl_nm * 1e-9)**2)
            recoil_nK = (recoil_J / kB) * 1e9
            
            # --- 顯示結果 (使用 st.metric 排版) ---
            st.subheader("📊 計算結果")
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            
            res_col1.metric("Trap Depth (阱深)", f"{trap_depth_uK:.2f} uK")
            res_col2.metric("Scattering Rate", f"{R_sc:.2e} Hz")
            res_col3.metric("Rayleigh Range", f"{z_R_mm:.2e} mm")
            res_col4.metric("Recoil Energy", f"{recoil_nK:.2f} nK")
            
        except Exception as e:
            st.error(f"計算錯誤: {e}\n請確認波長是否剛好在共振點（導致發散除以零）。")