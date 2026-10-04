import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap

# ---------------------------------------------------------
# Step 0: Ensure Required Directories
# ---------------------------------------------------------
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ---------------------------------------------------------
# Step 1: Data Ingestion & Chronological Sorting
# ---------------------------------------------------------
print("📥 Step 1: Loading Dataset...")
df = pd.read_csv('dengu_weather_dataset.csv')
df['start_date'] = pd.to_datetime(df['start_date'])
df = df.sort_values('start_date').reset_index(drop=True)

# ---------------------------------------------------------
# Step 2: Biological Feature Engineering
# ---------------------------------------------------------
print("⚙️ Step 2: Engineering Biological Lags & Cyclical Features...")

def generate_features(data):
    d = data.copy()
    
    # 1 to 4 Weeks Biological Mosquito/Incubation Lags
    for col in ['temp_mean', 'rainfall_total', 'humidity_mean', 'dengue_cases']:
        for lag in [1, 2, 3, 4]:
            d[f'{col}_lag_{lag}'] = d[col].shift(lag)
            
    # 2 & 4 Weeks Rolling Meteorological Averages
    for col in ['temp_mean', 'rainfall_total', 'humidity_mean']:
        d[f'{col}_roll_2'] = d[col].rolling(window=2).mean()
        d[f'{col}_roll_4'] = d[col].rolling(window=4).mean()
        
    # Trigonometric Cyclical Seasonality
    d['month'] = d['start_date'].dt.month
    d['sin_month'] = np.sin(2 * np.pi * d['month'] / 12)
    d['cos_month'] = np.cos(2 * np.pi * d['month'] / 12)
    
    return d

full_feat = generate_features(df)
full_feat = full_feat.dropna().reset_index(drop=True)

# Strict Chronological 80/20 Train-Test Split (Zero Data Leakage)
split_idx = int(len(full_feat) * 0.80)
train_feat = full_feat.iloc[:split_idx].reset_index(drop=True)
test_feat = full_feat.iloc[split_idx:].reset_index(drop=True)

ignore_cols = ['year', 'week', 'start_date', 'dengue_cases']
X_cols = [c for c in train_feat.columns if c not in ignore_cols]

# MinMaxScaler (Fitted strictly on training data)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(train_feat[X_cols])
X_test_scaled = scaler_X.transform(test_feat[X_cols])

y_train_scaled = scaler_y.fit_transform(train_feat[['dengue_cases']])
y_test_scaled = scaler_y.transform(test_feat[['dengue_cases']])

# Save Scalers & Feature Names
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')
joblib.dump(X_cols, 'models/feature_cols.pkl')

print(f"   Train Set Rows: {len(train_feat)} | Test Set Rows: {len(test_feat)}")
print(f"   Engineered Feature Dimensions: {len(X_cols)}")

# ---------------------------------------------------------
# Step 3: LSTM Temporal Feature Extractor
# ---------------------------------------------------------
print("\n🧠 Step 3: Training Optimized LSTM Feature Extractor...")
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

inputs = Input(shape=(1, X_train_scaled.shape[1]))
lstm_out = LSTM(64, return_sequences=False, name='lstm_embedding')(inputs)
dropout = Dropout(0.15)(lstm_out)
outputs = Dense(1)(dropout)

lstm_model = Model(inputs=inputs, outputs=outputs)
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='mse')
lstm_model.fit(X_train_lstm, y_train_scaled, epochs=80, batch_size=16, verbose=0)

embedding_model = Model(inputs=lstm_model.input, outputs=lstm_model.get_layer('lstm_embedding').output)
train_emb = embedding_model.predict(X_train_lstm, verbose=0)
test_emb = embedding_model.predict(X_test_lstm, verbose=0)

lstm_model.save('models/lstm_model.keras')

# ---------------------------------------------------------
# Step 4: Fine-Tuned XGBoost Regressor
# ---------------------------------------------------------
print("\n🌲 Step 4: Training Fine-Tuned XGBoost Regressor...")

X_train_xgb = np.hstack((X_train_scaled, train_emb))
X_test_xgb = np.hstack((X_test_scaled, test_emb))

xgb_model = XGBRegressor(
    n_estimators=180,
    learning_rate=0.035,
    max_depth=4,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.1,
    random_state=42
)

xgb_model.fit(X_train_xgb, y_train_scaled.ravel())
joblib.dump(xgb_model, 'models/xgb_model.pkl')

# ---------------------------------------------------------
# Step 5: Test Set Performance Evaluation
# ---------------------------------------------------------
pred_scaled = xgb_model.predict(X_test_xgb)
y_pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
y_true = scaler_y.inverse_transform(y_test_scaled).ravel()

r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print("\n" + "="*45)
print("✅ Model Evaluation Complete!")
print(f"   R² Score : {r2:.4f}")
print(f"   RMSE     : {rmse:.2f} cases")
print(f"   MAE      : {mae:.2f} cases")
print("="*45)

# ---------------------------------------------------------
# Step 6: TreeSHAP Feature Attribution Plot
# ---------------------------------------------------------
print("\n🔍 Generating TreeSHAP Summary Plots...")
lstm_emb_cols = [f'lstm_emb_{i}' for i in range(train_emb.shape[1])]
all_feature_names = X_cols + lstm_emb_cols

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_train_xgb)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_train_xgb, feature_names=all_feature_names, show=False)
plt.title("SHAP Feature Impact Summary Plot", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/shap_summary.png', dpi=300)
plt.close()

print("🎉 All pipeline steps executed successfully! Saved updated artifacts in /models & /outputs.")