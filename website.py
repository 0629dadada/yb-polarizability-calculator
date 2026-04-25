import streamlit as st
import numpy as np
import plotly.graph_objects as go
from polarizabilityyb1 import polarizability
from scattering_rate import scatterrate

st.set_page_config(page_title="Yb Physics Calculator", layout="wide")

# --- Helper Function: Convert Float to Fraction String ---
def to_fraction(val):
    """Converts 0.5 -> 1/2, 1.5 -> 3/2. Keeps integers as plain strings (no decimals)."""
    # 處理浮點數極微小誤差 (例如 0.00000)
    if abs(val) < 1e-9:
        return "0"
    # 偵測半整數 (例如 0.5, 1.5, -0.5)
    if abs(abs(val) % 1 - 0.5) < 1e-9:
        sign = "-" if val < 0 else ""
        return f"{sign}{int(abs(val) * 2)}/2"
    # 其他必定為整數，直接四捨五入後轉字串，完全剃除小數點
    return str(int(round(val)))

# --- Global Physical Constants ---
c_const = 299792458
h_const = 6.62607004e-34
hbar = h_const / (2 * np.pi)
kB = 1.380649e-23
epsilon_0 = 8.8541878128e-12
a0_const = 0.52917721067e-10
au_SI = 4 * np.pi * epsilon_0 * a0_const**3
m_Yb = 2.8733965E-25 

# Conversion factor from polarizability to V_ac/I (h Hz W^-1 cm^2)
conv_factor = (2 * np.pi * a0_const**3 / (c_const * h_const)) * 10000 

# --- Sidebar: Global Settings ---
st.sidebar.title("Global Parameters")

st.sidebar.markdown("### 1. Shared Physics Parameters")
# Display I as fractions in the selector
isotope = st.sidebar.selectbox(
    "Isotope", 
    [174, 171, 173],
    format_func=lambda x: f"{x} (I={to_fraction({174:0.0, 171:0.5, 173:2.5}[x])})"
)
I_dict = {174: 0.0, 171: 0.5, 173: 2.5}
I_val = I_dict[isotope]

pol_str = st.sidebar.selectbox("Polarization", 
                               ["pi (Linear)", "sigma+ (Circular)", "sigma- (Circular)"])
pol_dict = {"pi (Linear)": 0, "sigma+ (Circular)": 1, "sigma- (Circular)": -1}
p_val = pol_dict[pol_str]

# Dynamic help text for Beta Angle
if p_val == 0:
    beta_help_text = "For Linear (pi) polarization: Angle between the **POLARIZATION vector (e)** and the quantization axis (z-axis)."
else:
    beta_help_text = "For Circular (sigma) polarization: Angle between the **PROPAGATION vector (k)** and the quantization axis (z-axis)."

beta_angle = st.sidebar.number_input(
    "Incident Angle (Beta in deg)", 
    min_value=0.0, 
    max_value=180.0, 
    value=0.0,
    help=beta_help_text
)

st.sidebar.markdown("### 2. Plot Display Range (Plotter Only)")
col1, col2 = st.sidebar.columns(2)
with col1:
    min_wl = st.number_input("Min Wavelength (nm)", min_value=10.0, value=300.0, step=50.0)
with col2:
    max_wl = st.number_input("Max Wavelength (nm)", min_value=50.0, value=800.0, step=50.0)

unit_choice = st.sidebar.radio("Y-axis Unit", 
                               ["Atomic Unit (a.u.)", "Stark Shift (h Hz W^-1 cm^2)"])

# --- Main Page Tabs ---
tab1, tab2 = st.tabs(["📊 Polarizability Plotter", "🧮 Trap & Scattering Calculator"])

# ==========================================
# Tab 1: Polarizability Plotter
# ==========================================
with tab1:
    st.title("Ytterbium (Yb) Interactive Polarizability Plotter")
    st.markdown("Select states and mF values. The plot supports **hover values, scroll-to-zoom, and drag-to-pan**.")
    st.info("💡 **Physics Note:** A core polarizability correction of **-0.8 V_ac/I** is automatically applied to the 1S0 state.")

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
            F_val = info["J"] + I_val
            st.markdown(f"**{state_name} (F={to_fraction(F_val)})**")
            is_checked = st.checkbox(f"Show {state_name}", key=f"chk_{state_name}")
            
            if is_checked:
                mF_options = np.arange(-F_val, F_val + 1, 1.0)
                selected_mF = st.selectbox(
                    f"Select mF", 
                    mF_options, 
                    key=f"mf_{state_name}",
                    format_func=to_fraction  # Display mF as fractions without decimals
                )
                
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
        
        fig = go.Figure()
        
        for config in selected_configs:
            pol_val_array = polarizability(wavelengths, config["istate"], config["mF"], p_val, I_val, beta_angle)
            
            # --- APPLY 1S0 CORE CORRECTION ---
            if config["istate"] == 1:
                pol_val_array = pol_val_array - (0.8 / conv_factor)
            
            if unit_choice == "Atomic Unit (a.u.)":
                y_plot = -pol_val_array
                y_limit = 1000
                ylabel_str = "Polarizability (a.u.)"
            else:
                y_plot = pol_val_array * conv_factor
                y_limit = 40
                ylabel_str = "V_ac / I (h Hz W^-1 cm^2)"

            y_plot = np.array(y_plot, dtype=float)
            
            diffs = np.abs(np.diff(y_plot))
            jump_threshold = y_limit * 0.5
            jump_indices = np.where(diffs > jump_threshold)[0]
            
            for idx in jump_indices:
                y_plot[idx] = np.nan
                y_plot[idx+1] = np.nan
                
            label_str = f"{config['name']}, F={to_fraction(config['J']+I_val)}, mF={to_fraction(config['mF'])}"
            
            fig.add_trace(go.Scatter(
                x=wavelengths, 
                y=y_plot, 
                mode='lines', 
                name=label_str,
                hovertemplate="Wavelength: %{x:.2f} nm<br>Value: %{y:.2f}<extra></extra>"
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
        st.info("Please select at least one state above to display the plot.")


# ==========================================
# Tab 2: Trap & Scattering Calculator 
# ==========================================
with tab2:
    st.title("Optical Trap Depth & Scattering Rate Calculator (1S0 Ground State)")
    
    transitions_data = {
        "(6s6p)3P1 (≈ 556 nm)": {"wn": 17992.00699, "gamma_khz": 182.2},
        "(6s6p)1P1 (≈ 399 nm)": {"wn": 25068.222, "gamma_khz": 29000.0}
    }

    # --- UI Input Section ---
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.subheader("1. Light Source & Laser Settings")
        regime = st.radio("Detuning Regime", ["Far-Detuned", "Near Resonance"])
        
        if regime == "Far-Detuned":
            input_wvl = st.number_input("Input Wavelength (nm)", value=532.0, step=1.0, format="%.4f")
            eval_wvl_nm = input_wvl
        else:
            selected_trans = st.selectbox("Select Near-Resonant Transition", list(transitions_data.keys()))
            detuning_mhz = st.number_input("Detuning (MHz)", value=-10000.0, step=10.0)
            
            gamma_khz = transitions_data[selected_trans]["gamma_khz"]
            
            wn = transitions_data[selected_trans]["wn"]
            f_res = c_const * wn * 100 
            f_eval = f_res + (detuning_mhz * 1e6)
            eval_wvl_nm = (c_const / f_eval) * 1e9
            
            st.caption(f"Evaluated Wavelength: **{eval_wvl_nm:.6f} nm**")
            st.caption(f"Auto-applied Natural Linewidth Γ/2π: **{gamma_khz} kHz**")
            
    with col_input2:
        st.subheader("2. Trap Geometry & Power")
        trap_type = st.radio("Trap Type", ["Dipole Trap (Single Beam)", "Conveyor Belt (1D Lattice)"])
        waist_um = st.number_input("Beam Waist (um)", value=200.0, step=10.0)
        power_mw = st.number_input("Power (mW)", value=100.0, step=10.0, format="%.3f")
        
    st.divider()

    # --- Core Calculations ---
    if st.button("🚀 Calculate", use_container_width=True):
        try:
            # 1. Calculate Intensity
            w0_m = waist_um * 1e-6
            P_W = power_mw * 1e-3
            I_0 = 2 * P_W / (np.pi * w0_m**2)
            
            if trap_type == "Conveyor Belt (1D Lattice)":
                I_calc = 4 * I_0
                st.info("ℹ️ 1D Lattice: Calculation uses antinode peak intensity (I = 4 × I_0)")
            else:
                I_calc = I_0
                st.info("ℹ️ Dipole Trap: Calculation uses focal center intensity (I = I_0)")

            # 2. Calculate Trap Depth (Applying -0.8 V_ac/I correction to 1S0)
            raw_pol_array = polarizability(eval_wvl_nm, 1, 0, p_val, I_val, beta_angle)
            corrected_pol_array = raw_pol_array - (0.8 / conv_factor)
            
            alpha_au_val = abs(corrected_pol_array)
            alpha_SI = alpha_au_val * au_SI
            U_joule = (alpha_SI * I_calc) / (2 * epsilon_0 * c_const)
            trap_depth_uK = (U_joule / kB) * 1e6

            # 3. Calculate Scattering Rate
            if regime == "Near Resonance":
                st.success("✅ Applied two-level approximation saturated scattering formula (Lorentzian Profile)")
                lambda_res_m = 1.0 / (transitions_data[selected_trans]["wn"] * 100)
                gamma_rad = 2 * np.pi * gamma_khz * 1e3
                delta_rad = 2 * np.pi * detuning_mhz * 1e6
                
                Isat = (np.pi * h_const * c_const * gamma_rad) / (3 * lambda_res_m**3)
                s0 = I_calc / Isat
                
                R_sc = (gamma_rad / 2) * (s0 / (1 + s0 + 4 * (delta_rad / gamma_rad)**2))
            else:
                st.success("✅ Applied multi-level Kramers-Heisenberg far-detuned formula")
                rate_per_I = scatterrate(eval_wvl_nm, 1)
                R_sc = float(rate_per_I) * I_calc

            # 4. Calculate Rayleigh Range & Recoil Energy
            z_R_mm = (np.pi * w0_m**2 / (eval_wvl_nm * 1e-9)) * 1e3
            recoil_J = h_const**2 / (2 * m_Yb * (eval_wvl_nm * 1e-9)**2)
            recoil_nK = (recoil_J / kB) * 1e9
            
            # --- Display Results ---
            st.subheader("📊 Calculation Results (with -0.8 V_ac/I Core Correction)")
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            
            res_col1.metric("Trap Depth", f"{trap_depth_uK:.2f} uK")
            res_col2.metric("Scattering Rate", f"{R_sc:.2e} Hz")
            res_col3.metric("Rayleigh Range", f"{z_R_mm:.2e} mm")
            res_col4.metric("Recoil Energy", f"{recoil_nK:.2f} nK")
            
        except Exception as e:
            st.error(f"Calculation Error: {e}\nPlease check if the input parameters are valid.")
