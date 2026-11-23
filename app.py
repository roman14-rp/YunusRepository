import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model_pinjaman.pkl")

st.title("Prediksi Kelayakan Pinjaman Bank")
st.write("Isi data pemohon, lalu klik tombol Prediksi.")

col1, col2 = st.columns(2)
with col1:
    Gender = st.selectbox("Jenis kelamin", ["Male", "Female"])
    Married = st.selectbox("Sudah menikah?", ["Yes", "No"])
    Dependents = st.selectbox("Jumlah tanggungan", ["0", "1", "2", "3+"])
    Education = st.selectbox("Pendidikan", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Wiraswasta?", ["No", "Yes"])
    Property_Area = st.selectbox("Lokasi properti", ["Urban", "Semiurban", "Rural"])
with col2:
    ApplicantIncome = st.number_input("Pendapatan pemohon", min_value=0, value=5000)
    CoapplicantIncome = st.number_input("Pendapatan pendamping", min_value=0, value=0)
    LoanAmount = st.number_input("Jumlah pinjaman (dalam ribuan)", min_value=1, value=130)
    Loan_Amount_Term = st.selectbox("Tenor pinjaman (bulan)", [360, 180, 480, 300, 240, 120, 84, 60, 36, 12])
    Credit_History = st.selectbox("Riwayat kredit baik? (1 = ya, 0 = tidak)", [1.0, 0.0])

if st.button("Prediksi"):
    total = ApplicantIncome + CoapplicantIncome
    data = pd.DataFrame([{
        "Credit_History": Credit_History,
        "Loan_Amount_Term": Loan_Amount_Term,
        "LoanAmount_Log": np.log1p(LoanAmount),
        "Income_Log": np.log1p(total),
        "Rasio_Pinjaman": LoanAmount / total if total > 0 else 0,
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "Property_Area": Property_Area,
    }])

    hasil = model.predict(data)[0]
    peluang = model.predict_proba(data)[0][1]

    if hasil == 1:
        st.success(f"Pinjaman diprediksi DISETUJUI (peluang {peluang:.0%})")
    else:
        st.error(f"Pinjaman diprediksi DITOLAK (peluang disetujui {peluang:.0%})")
