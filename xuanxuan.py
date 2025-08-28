import matplotlib.pyplot as plt
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap

# 加载随机森林模型
model = joblib.load('E:/RS/wangxuanxuan/结果/特征筛选/RF/白血病三分类1.pkl')

# 定义特征名称
feature_names = [
    "B%", "E", "B", "M", "E%", "N", "L", "M%",
    "RBC", "WBC", "L%", "PCT", "PLT", "N%",
    "PDW", "MCH", "RDWSD", "MCV"
]

# 类别名称映射
class_names = {0: "APL/M3", 1: "AMoL/M5", 2: "ALL"}

# Streamlit 用户界
st.title("Acute Leukaemia Subtype Classification Model")

# 用户输入特征数据（加上单位说明）
B_percent   = st.number_input("B% (%):", min_value=0.0, max_value=100.0, value=2.0)
E           = st.number_input("E (×10⁹/L):", min_value=0.0, max_value=10.0, value=0.1)
B           = st.number_input("B (×10⁹/L):", min_value=0.0, max_value=10.0, value=0.2)
M           = st.number_input("M (×10⁹/L):", min_value=0.0, max_value=10.0, value=0.3)
E_percent   = st.number_input("E% (%):", min_value=0.0, max_value=100.0, value=1.5)
N           = st.number_input("N (×10⁹/L):", min_value=0.0, max_value=20.0, value=4.0)
L           = st.number_input("L (×10⁹/L):", min_value=0.0, max_value=10.0, value=2.0)
M_percent   = st.number_input("M% (%):", min_value=0.0, max_value=100.0, value=5.0)
RBC         = st.number_input("RBC (×10¹²/L):", min_value=0.0, max_value=10.0, value=4.5)
WBC         = st.number_input("WBC (×10⁹/L):", min_value=0.0, max_value=50.0, value=7.0)
L_percent   = st.number_input("L% (%):", min_value=0.0, max_value=100.0, value=25.0)
PCT         = st.number_input("PCT (%):", min_value=0.0, max_value=5.0, value=0.2)
PLT         = st.number_input("PLT (×10⁹/L):", min_value=0.0, max_value=1000.0, value=250.0)
N_percent   = st.number_input("N% (%):", min_value=0.0, max_value=100.0, value=65.0)
PDW         = st.number_input("PDW (fL):", min_value=0.0, max_value=30.0, value=12.0)
MCH         = st.number_input("MCH (pg):", min_value=0.0, max_value=100.0, value=29.0)
RDWSD       = st.number_input("RDWSD (fL):", min_value=0.0, max_value=100.0, value=42.0)
MCV         = st.number_input("MCV (fL):", min_value=0.0, max_value=120.0, value=90.0)

# 转化为模型输入格式
feature_values = [
    B_percent, E, B, M, E_percent, N, L, M_percent,
    RBC, WBC, L_percent, PCT, PLT, N_percent,
    PDW, MCH, RDWSD, MCV
]
features = np.array([feature_values])

# 点击按钮进行预测
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果（映射类别名）
    st.write(f"**Predicted Class:** {predicted_class} → {class_names[predicted_class]}")
    st.write("**Prediction Probabilities:**")
    for i, prob in enumerate(predicted_proba):
        st.write(f" - {class_names[i]}: {prob:.3f}")