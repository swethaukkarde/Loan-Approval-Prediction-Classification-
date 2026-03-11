import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# Load model
model = joblib.load("loan_model.pkl")

# Custom CSS for styling
st.markdown("""
<style>
.main-title {
    font-size:40px;
    font-weight:bold;
    color:#2c3e50;
}

.subtitle {
    font-size:18px;
    color:gray;
}

.stButton>button {
    background-color:#2ecc71;
    color:white;
    font-size:18px;
    border-radius:10px;
    padding:10px 20px;
}

.result-success {
    background-color:#d4edda;
    padding:20px;
    border-radius:10px;
    font-size:22px;
    color:#155724;
}

.result-fail {
    background-color:#f8d7da;
    padding:20px;
    border-radius:10px;
    font-size:22px;
    color:#721c24;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-title">🏦 Loan Approval Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI system to predict loan approval based on applicant financial details.</p>', unsafe_allow_html=True)

st.divider()

# Sidebar Inputs
st.sidebar.header("Applicant Information")

gender = st.sidebar.selectbox("Gender", ["Male","Female"])
married = st.sidebar.selectbox("Married", ["Yes","No"])
dependents = st.sidebar.selectbox("Dependents", ["0","1","2","3+"])
education = st.sidebar.selectbox("Education", ["Graduate","Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes","No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban","Semiurban","Rural"])

st.sidebar.header("Financial Details")

applicant_income = st.sidebar.number_input("Applicant Income",0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income",0)
loan_amount = st.sidebar.number_input("Loan Amount",0)
loan_term = st.sidebar.number_input("Loan Term",0)
credit_history = st.sidebar.selectbox("Credit History",[1.0,0.0])

# Encoding
gender = 1 if gender=="Male" else 0
married = 1 if married=="Yes" else 0
education = 0 if education=="Graduate" else 1
self_employed = 1 if self_employed=="Yes" else 0

if dependents=="3+":
    dependents=3
else:
    dependents=int(dependents)

area_map={
"Urban":2,
"Semiurban":1,
"Rural":0
}

property_area=area_map[property_area]

# Feature Engineering
total_income = applicant_income + coapplicant_income

loan_income_ratio = loan_amount/total_income if total_income!=0 else 0

input_data = np.array([[gender,
                        married,
                        dependents,
                        education,
                        self_employed,
                        applicant_income,
                        coapplicant_income,
                        loan_amount,
                        loan_term,
                        credit_history,
                        property_area,
                        total_income,
                        loan_income_ratio]])

col1,col2 = st.columns([1,1])

with col1:
    st.subheader("Prediction")

    if st.button("Predict Loan Status"):

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        if prediction[0]==1:
            st.markdown(
            f'<div class="result-success">✅ Loan Approved<br>Approval Probability: {probability[0][1]*100:.2f}%</div>',
            unsafe_allow_html=True)

        else:
            st.markdown(
            f'<div class="result-fail">❌ Loan Rejected<br>Approval Probability: {probability[0][1]*100:.2f}%</div>',
            unsafe_allow_html=True)

with col2:
    st.subheader("About this Model")

    st.info("""
This Machine Learning model predicts whether a loan application will be approved.

Features used:
- Applicant Income
- Coapplicant Income
- Loan Amount
- Credit History
- Education
- Marital Status
- Property Area

Model Type:
Random Forest / Gradient Boosting
""")

st.divider()

st.caption("Built with ❤️ using Streamlit & Scikit-learn")
