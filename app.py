import streamlit as st
import yaml

from app.loader import load_models
from app.utils import build_applicant_from_dict
from app.predict import two_stage_predict

with open('config.yaml') as f:
    config=yaml.safe_load(f)

st.set_page_config(page_title='Loan Approval',layout='centered')
st.title("Loan Approval - Two stage predictor")

cls,reg=load_models(config)

st.sidebar.header("Model Info")
try:
    st.sidebar.write("Classifier Excepts : ",list(cls.feature_names_in_))
except:
    st.sidebar.write("Classifier feature names unavailable")

st.header("Applicant Details")
default=config['ui']['default_inputs']

cols=st.columns(2)
with cols[0]:
    no_of_dependents=st.number_input("No of Dependents",value=int(default['no_of_dependents']))
    education=st.selectbox('Education',options=['Graduate','Not Graduate'],index=0 if default['education']=='Graduate' else 1)
    self_employed=st.selectbox("Self employed",options=['Yes','No'],index=0 if default['education']=='Yes' else 1)
    income_annum=st.number_input("Annual Income",value=float(default['income_annum']))
    loan_amount=st.number_input("Loan Amount Requested",value=float(default['loan_amount']))

with cols[1]:
    loan_term=st.number_input("Loan Term (Years)",value=int(default['loan_term']))
    cibil_score=st.number_input("CIBIL Score ",value=int(default['cibil_score']))
    residental_assets_value=st.number_input("Residental assets",value=float(default['residential_assets_value']))
    commercial_assets_value=st.number_input("Commercial assets",value=float(default['commercial_assets_value']))
    luxury_assets_value=st.number_input("Luxury assets",value=float(default['luxury_assets_value']))
    bank_asset_value=st.number_input("Bank assets",value=float(default['bank_asset_value']))


applicant={
    'no_of_dependents': no_of_dependents,
    'education' : education,
    'self_employed' : self_employed,
    'income_annum' : income_annum,
    'loan_amount' : loan_amount,
    'loan_term' : loan_term,
    'cibil_score' : cibil_score,
    'residential_assets_value' : residental_assets_value,
    'commercial_assets_value' : commercial_assets_value,
    'luxury_assets_value' : luxury_assets_value,
    'bank_asset_value' : bank_asset_value
}


if st.button("Predict"):
    try:
        excpected_cols=list(cls.feature_names_in_)
        applicant_df=build_applicant_from_dict(applicant,excpected_cols)
        results=two_stage_predict(cls,reg,applicant_df)
        res=results[0]

        if res['approved']==1:
            st.success("✅ Approved")
            st.write(f"**Confidence:** {res['approved_proba']:.2%}")
            st.write(f"**Predicted Loan Amount:** ₹{res['reg_pred']:,.2f}")
        else:
            st.error("❌ Rejected")
            st.write(f"**Approval Probability:** {res['approved_proba']:.2%}")
            st.write(f"**Rejection Probability:** {res['rejected_proba']:.2%}")
    except Exception as e:
        st.error(f"Predicted failed : {e}")