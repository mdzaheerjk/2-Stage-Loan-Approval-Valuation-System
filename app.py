import streamlit as st
import yaml

from app.loader import load_models
from app.utils import build_application_from_dict
from app.predict import two_stage_predict

# -------------------------------
# Load Configuration
# -------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Loan Approval - Two Stage Predictor")
st.markdown("Enter the applicant details below and click **Predict**.")

# -------------------------------
# Load Models
# -------------------------------
try:
    cls, reg = load_models(config)
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("📋 Model Information")

try:
    st.sidebar.write("Classifier Features:")
    st.sidebar.write(list(cls.feature_names_in_))
except Exception:
    st.sidebar.warning("Classifier feature names unavailable.")

# -------------------------------
# Default Inputs
# -------------------------------
default = config["ui"]["default_inputs"]

st.header("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        value=int(default["no_of_dependents"]),
        step=1
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"],
        index=0 if default["education"] == "Graduate" else 1
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"],
        index=0 if default["self_employed"] == "Yes" else 1
    )

    income_annum = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=float(default["income_annum"]),
        step=10000.0
    )

    loan_amount = st.number_input(
        "Loan Amount Requested",
        min_value=0.0,
        value=float(default["loan_amount"]),
        step=10000.0
    )

with col2:
    loan_term = st.number_input(
        "Loan Term",
        min_value=1,
        value=int(default["loan_term"]),
        step=1
    )

    cibil_score = st.number_input(
        "CIBIL Score",
        min_value=300,
        max_value=900,
        value=int(default["cibil_score"]),
        step=1
    )

    residential_assets_value = st.number_input(
        "Residential Assets Value",
        min_value=0.0,
        value=float(default["residential_assets_value"]),
        step=10000.0
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets Value",
        min_value=0.0,
        value=float(default["commercial_assets_value"]),
        step=10000.0
    )

    luxury_assets_value = st.number_input(
        "Luxury Assets Value",
        min_value=0.0,
        value=float(default["luxury_assets_value"]),
        step=10000.0
    )

    bank_asset_value = st.number_input(
        "Bank Assets Value",
        min_value=0.0,
        value=float(default["bank_asset_value"]),
        step=10000.0
    )

# -------------------------------
# Applicant Dictionary
# -------------------------------
applicant = {
    "no_of_dependents": no_of_dependents,
    "education": education,
    "self_employed": self_employed,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": residential_assets_value,
    "commercial_assets_value": commercial_assets_value,
    "luxury_assets_value": luxury_assets_value,
    "bank_asset_value": bank_asset_value,
}

# -------------------------------
# Predict Button
# -------------------------------
if st.button("🔍 Predict", use_container_width=True):

    with st.spinner("Making prediction..."):

        try:
            expected_cols = list(cls.feature_names_in_)

            applicant_df = build_application_from_dict(
                applicant,
                expected_cols
            )

            results = two_stage_predict(
                cls,
                reg,
                applicant_df
            )

            res = results[0]

            st.divider()
            st.subheader("Prediction Results")

            st.metric(
                "Approval Probability",
                f"{res['approved_prob']:.2%}"
            )

            st.progress(float(res["approved_prob"]))

            if res["approved"] == 1:
                st.success("✅ Loan Approved")

                if res.get("reg_pred") is not None:
                    st.metric(
                        "Predicted Loan Amount",
                        f"₹{res['reg_pred']:,.2f}"
                    )

            else:
                st.error("❌ Loan Rejected")

            with st.expander("View Input Data"):
                st.dataframe(applicant_df)

        except Exception as e:
            st.error(f"❌ Prediction Failed: {e}")