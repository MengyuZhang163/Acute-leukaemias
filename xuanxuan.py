import streamlit as st
import joblib
import numpy as np

# 加载模型
model = joblib.load('model.pkl')

# 特征名称（保持顺序与模型一致）
feature_names = [
    "B%", "E", "B", "M", "E%", "N", "L", "M%",
    "RBC", "WBC", "L%", "PCT", "PLT", "N%",
    "PDW", "MCH", "RDWSD", "MCV"
]

# 类别映射
class_names = {0: "APL/M3", 1: "AMoL/M5", 2: "ALL"}

st.title("Acute Leukaemia Subtype Classification Model")

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    B_percent = st.number_input("B% (%):", min_value=0.0, max_value=100.0, value=2.0, key="B%")
    E = st.number_input("E (×10⁹/L):", min_value=0.0, max_value=10.0, value=0.1, key="E")
    B = st.number_input("B (×10⁹/L):", min_value=0.0, max_value=10.0, value=0.2, key="B")
    M = st.number_input("M (×10⁹/L):", min_value=0.0, max_value=10.0, value=0.3, key="M")
    E_percent = st.number_input("E% (%):", min_value=0.0, max_value=100.0, value=1.5, key="E%")
    N = st.number_input("N (×10⁹/L):", min_value=0.0, max_value=20.0, value=4.0, key="N")
    L = st.number_input("L (×10⁹/L):", min_value=0.0, max_value=10.0, value=2.0, key="L")
    M_percent = st.number_input("M% (%):", min_value=0.0, max_value=100.0, value=5.0, key="M%")
    RBC = st.number_input("RBC (×10¹²/L):", min_value=0.0, max_value=10.0, value=4.5, key="RBC")

with col2:
    WBC = st.number_input("WBC (×10⁹/L):", min_value=0.0, max_value=50.0, value=7.0, key="WBC")
    L_percent = st.number_input("L% (%):", min_value=0.0, max_value=100.0, value=25.0, key="L%")
    PCT = st.number_input("PCT (%):", min_value=0.0, max_value=5.0, value=0.2, key="PCT")
    PLT = st.number_input("PLT (×10⁹/L):", min_value=0.0, max_value=1000.0, value=250.0, key="PLT")
    N_percent = st.number_input("N% (%):", min_value=0.0, max_value=100.0, value=65.0, key="N%")
    PDW = st.number_input("PDW (fL):", min_value=0.0, max_value=30.0, value=12.0, key="PDW")
    MCH = st.number_input("MCH (pg):", min_value=0.0, max_value=100.0, value=29.0, key="MCH")
    RDWSD = st.number_input("RDWSD (fL):", min_value=0.0, max_value=100.0, value=42.0, key="RDWSD")
    MCV = st.number_input("MCV (fL):", min_value=0.0, max_value=120.0, value=90.0, key="MCV")

# 转化为模型输入格式（保持与 feature_names 一致的顺序）
feature_values = [
    B_percent, E, B, M, E_percent, N, L, M_percent,
    RBC, WBC, L_percent, PCT, PLT, N_percent,
    PDW, MCH, RDWSD, MCV
]
features = np.array([feature_values])

# 预测按钮（放在两列下方）
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示结果
    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {predicted_class} → {class_names[predicted_class]}")

    st.subheader("Prediction Probabilities")
    for i, prob in enumerate(predicted_proba):
        st.write(f"- {class_names[i]}: {prob:.3f}")






