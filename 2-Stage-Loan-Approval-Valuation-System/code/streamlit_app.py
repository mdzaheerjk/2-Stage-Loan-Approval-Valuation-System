import streamlit as st
import yaml

from app.loader import load_models
from app.predict import two_stage_predict
from app.utlis import build_applicant_form_dict

with open('config.yaml') as f:
    config=yaml.safe_load(f)

cls,reg=load_models(config)

st.set_page_config("Loan Approval",layout='centered')
st.title("Loan Approval - Two Stage Predictor")



st.sidebar.header("Model Info")
try:
    pass
except Exception:
    st.sidebar.write("Classifier Features Names not Available")

st.header("Applicant Details")
default=config['ui']['default_input']

cols=st.columns(2)
with cols[0]:
    no_of_dependents=st.number_input("No of Dependents",value=int(default['no_of_dependents']))
    education=st.selectbox("Education",options=['Graduate','Not Graduate'],index=0 if default['education']=='Graduate' else 1)
    self_employed=st.selectbox('Self Employed',options=['Yes','No'],index=0 if default['self_employed']=='Yes' else 0)
    income_annum=st.number_input("Annual Income",value=float(default['income_annum']))
    loan_amount=st.number_input("Loan Ammount Requested",value=float(default['loan_amount']))
with cols[1]:
    loan_term=st.number_input("loan Term (years)",value=int(default['loan_term']))
    cibil_score=st.number_input("cibil score",value=int(default['cibil_score']))
    residential_assets_value=st.number_input("residential assets value",value=float(default['residential_assets_value']))
    commercial_assets_value=st.number_input("commercial assets value",value=float(default['commercial_assets_value']))
    luxury_assets_value=st.number_input("luxury assets value",value=float(default['luxury_assets_value']))
    bank_asset_value=st.number_input("bank asset value",value=float(default['bank_asset_value']))


if st.button("Predict"):
    try:
        pass
    except Exception as e:
        st.error(f"Predection Failed {e}")
