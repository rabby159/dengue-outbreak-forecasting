import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ---------------------------------------------------------
# Page Config & Custom Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dengue Forecasting System | GUB CSE Thesis",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 { color: #ffffff; margin-bottom: 4px; font-size: 24px; }
    .main-header h3 { color: #e0e0e0; margin-top: 0px; font-size: 15px; font-weight: 400; }
    .meta-badge {
        background-color: rgba(255, 255, 255, 0.15);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        margin-right: 8px;
        display: inline-block;
    }
    .risk-high {
        background: rgba(231, 76, 60, 0.15);
        border: 2px solid #e74c3c;
        padding: 20px;
        border-radius: 10px;
        color: #ff6b6b;
    }
    .risk-mod {
        background: rgba(241, 196, 15, 0.15);
        border: 2px solid #f1c40f;
        padding: 20px;
        border-radius: 10px;
        color: #f39c12;
    }
    .risk-low {
        background: rgba(46, 204, 113, 0.15);
        border: 2px solid #2ecc71;
        padding: 20px;
        border-radius: 10px;
        color: #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-header">
    <span class="meta-badge">🏛️ Green University of Bangladesh</span>
    <span class="meta-badge">💻 Department of Computer Science & Engineering</span>
    <span class="meta-badge">🎓 Capstone Thesis Project</span>
    <h1>An Explainable AI (XAI) Enabled Hybrid LSTM-XGBoost Framework for Dengue Outbreak Forecasting</h1>
    <h3>Early Warning Decision Support System for Public Health Surveillance</h3>
</div>
""", unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_artifacts():
    lstm = tf.keras.models.load_model('models/lstm_model.keras')
    xgb = joblib.load('models/xgb_model.pkl')
    s_x = joblib.load('models/scaler_X.pkl')
    s_y = joblib.load('models/scaler_y.pkl')
    f_cols = joblib.load('models/feature_cols.pkl')
    return lstm, xgb, s_x, s_y, f_cols

try:
    lstm_model, xgb_model, scaler_X, scaler_y, feature_cols = load_artifacts()
except Exception as e:
    st.error(f"⚠️️ Model artifacts load failed! Make sure `train_pipeline.py` ran successfully. Error: {e}")
    st.stop()

# Sidebar Setup
st.sidebar.title("🎛️ Live Parameter Controls")

st.sidebar.markdown("### 🌤️ Meteorological Factors")
temp_input = st.sidebar.number_input("Mean Temperature (°C)", min_value=15.0, max_value=35.0, value=28.95, step=0.1)
rain_input = st.sidebar.number_input("Total Rainfall (mm)", min_value=0.0, max_value=300.0, value=44.15, step=1.0)
hum_input = st.sidebar.number_input("Mean Humidity (%)", min_value=40.0, max_value=100.0, value=77.02, step=0.1)
month_input = st.sidebar.selectbox("Month of Year", options=list(range(1, 13)), index=8, format_func=lambda x: f"Month {x}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Epidemic Baseline")
recent_cases_input = st.sidebar.number_input("Recent Weekly Cases (Last Week)", min_value=0, max_value=25000, value=10827, step=100)

st.sidebar.markdown("---")
with st.sidebar.expander("👥 Research Team & Supervision", expanded=False):
    st.markdown("""
    **Supervisor:**
    * **Babe Sultana**
      *Lecturer, Dept. of CSE, GUB*

    **Authors / Research Team:**
    1. **Md Rabby** (ID: 213902124)
    2. **Gazi Faria Akter** (ID: 213902067)
    3. **Md Julfikar Alam** (ID: 223902025)
    """)

# Feature Pipeline
input_dict = {col: 0.0 for col in feature_cols}
input_dict['temp_mean'] = float(temp_input)
input_dict['rainfall_total'] = float(rain_input)
input_dict['humidity_mean'] = float(hum_input)

for lag in [1, 2, 3, 4]:
    if f'temp_mean_lag_{lag}' in input_dict: input_dict[f'temp_mean_lag_{lag}'] = float(temp_input)
    if f'rainfall_total_lag_{lag}' in input_dict: input_dict[f'rainfall_total_lag_{lag}'] = float(rain_input)
    if f'humidity_mean_lag_{lag}' in input_dict: input_dict[f'humidity_mean_lag_{lag}'] = float(hum_input)
    if f'dengue_cases_lag_{lag}' in input_dict: input_dict[f'dengue_cases_lag_{lag}'] = float(recent_cases_input)

for r in [2, 4]:
    if f'temp_mean_roll_{r}' in input_dict: input_dict[f'temp_mean_roll_{r}'] = float(temp_input)
    if f'rainfall_total_roll_{r}' in input_dict: input_dict[f'rainfall_total_roll_{r}'] = float(rain_input)
    if f'humidity_mean_roll_{r}' in input_dict: input_dict[f'humidity_mean_roll_{r}'] = float(hum_input)

input_dict['month'] = float(month_input)
input_dict['sin_month'] = np.sin(2 * np.pi * month_input / 12)
input_dict['cos_month'] = np.cos(2 * np.pi * month_input / 12)

input_df = pd.DataFrame([input_dict])[feature_cols]
X_scaled = scaler_X.transform(input_df)

X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
embedding_model = tf.keras.models.Model(inputs=lstm_model.input, outputs=lstm_model.get_layer('lstm_embedding').output)
lstm_emb = embedding_model.predict(X_lstm, verbose=0)

X_xgb = np.hstack((X_scaled, lstm_emb))
pred_scaled = xgb_model.predict(X_xgb)
predicted_cases = max(0, int(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]))

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Live Forecast & Warning", "📊 Parameter Analytics", "🎓 Thesis & Architecture Details"])

with tab1:
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("### 📊 Next Week Dengue Forecast")
        st.metric(
            label="Predicted Dengue Cases (Next 7 Days)",
            value=f"{predicted_cases:,} Cases",
            delta=f"{predicted_cases - recent_cases_input:+,} vs Last Week",
            delta_color="inverse"
        )
        st.caption("Forecast computed via Hybrid LSTM Temporal Embedding + XGBoost Gradient Boosting.")
        
    with c2:
        st.markdown("### 🚨 Early Warning Decision Level")
        if predicted_cases > 3000:
            st.markdown("""
            <div class="risk-high">
                <h2>🚨 HIGH OUTBREAK RISK</h2>
                <p><b>Epidemic Threshold Cross Detected (> 3,000 Cases).</b></p>
                <hr>
                <p><b>Recommended Actions:</b></p>
                <ul>
                    <li>Immediate Emergency Vector Control & Adulticide Fogging in High-Density Zones.</li>
                    <li>Hospital Resource Allocation: Dedicated ICU/Dengue Ward Expansion.</li>
                    <li>Activate Media & Citizen Warning Bulletins.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif predicted_cases > 1000:
            st.markdown("""
            <div class="risk-mod">
                <h2>⚠️ MODERATE OUTBREAK RISK</h2>
                <p><b>Moderate Transmission Potential (1,000 – 3,000 Cases).</b></p>
                <hr>
                <p><b>Recommended Actions:</b></p>
                <ul>
                    <li>Enhanced Larvicidal Operations in Stagnant Water Pockets.</li>
                    <li>Community Awareness Drives & Household Container Inspections.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-low">
                <h2>✅ LOW OUTBREAK RISK</h2>
                <p><b>Baseline Epidemiological Activity (< 1,000 Cases).</b></p>
                <hr>
                <p><b>Recommended Actions:</b></p>
                <ul>
                    <li>Standard Routine Entomological Surveillance.</li>
                    <li>Regular Climate Parameter Monitoring.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.subheader("📋 Input Parameter Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", f"{temp_input:.1f} °C")
    col2.metric("Rainfall", f"{rain_input:.1f} mm")
    col3.metric("Humidity", f"{hum_input:.1f} %")
    col4.metric("Last Week Cases", f"{recent_cases_input:,}")
    
    st.markdown("---")
    st.subheader("🦟 Mosquito Breeding Suitability Index")
    
    temp_score = 100 if 25 <= temp_input <= 30 else (50 if 20 <= temp_input <= 35 else 20)
    rain_score = 100 if rain_input >= 50 else (rain_input * 2)
    hum_score = 100 if hum_input >= 75 else (hum_input)
    suitability = int((temp_score + rain_score + hum_score) / 3)
    
    st.progress(suitability / 100)
    st.caption(f"Estimated Vector Breeding Suitability Score: **{suitability}%** (Optimal Aedes EIP Range: 28-30°C, Humidity >75%)")

with tab3:
    st.subheader("🎓 Academic & Technical Overview")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        #### 📌 Model Evaluation Summary (Unseen Test Data)
        * **$R^2$ Accuracy Score:** `0.9346` (93.46% Variance Explained)
        * **Root Mean Squared Error (RMSE):** `666.83` Cases
        * **Mean Absolute Error (MAE):** `501.75` Cases
        * **Training Period:** 2018 – 2024 Weekly Records
        * **Test Period (Unseen):** 2024 – 2026 Records
        """)
        
    with col_b:
        st.markdown("""
        #### 🛠️ Core Innovation & Architecture
        1. **Biological Lag Engineering:** 2-4 week delayed meteorological variables encoding Aedes mosquito EIP/IIP incubation periods.
        2. **Trigonometric Cyclical Encoding:** $\sin/\cos$ month transformation preserving continuous seasonal loop.
        3. **Hybrid Synergy:** 64-dim LSTM latent sequence embeddings fused with XGBoost decision boundaries.
        4. **Explainable AI (TreeSHAP):** Game-theoretic feature attribution.
        """)
        
    st.markdown("---")
    st.subheader("👥 Project Metadata")
    st.markdown("""
    * **Institution:** Department of Computer Science & Engineering, Green University of Bangladesh
    * **Supervisor:** Babe Sultana, Lecturer, Dept. of CSE, GUB
    * **Research Team:** Md Rabby (213902124), Gazi Faria Akter (213902067), Md Julfikar Alam (223902025)
    """)
