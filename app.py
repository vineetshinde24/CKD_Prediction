
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #168A86;
    }
    .subtitle {
        color: #64748b;
        font-size: 17px;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #e0f2fe;
        color: #0f172a;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__), "ckd_model.pkl"
    )

    saved_data = joblib.load(model_path)

    return saved_data["model"], saved_data["features"]

try:
    model, features = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🩺 CKD Prediction")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict CKD", "Model Information"]
)

st.sidebar.divider()
st.sidebar.caption("AI Healthcare Mini Project")
st.sidebar.caption("Random Forest Classifier")

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":

    st.markdown(
        '<div class="main-title">Chronic Kidney Disease Prediction System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered healthcare prediction using Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This application demonstrates CKD classification "
        "using a trained Random Forest model."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Records", "400")

    with col2:
        st.metric("Input Features", len(features))

    with col3:
        st.metric("ML Algorithm", "Random Forest")

    st.divider()

    st.subheader("About Chronic Kidney Disease")

    st.write("""
    Chronic Kidney Disease (CKD) is a long-term condition
    in which the kidneys gradually lose their ability
    to filter waste and excess fluids from the blood.

    Early detection and appropriate medical care can help
    manage CKD and reduce complications.

    This project uses machine learning to classify
    patient measurements as CKD or No CKD.
    """)

    st.subheader("Application Features")

    st.write("""
    - Enter clinical measurements using a simple form.
    - Predict CKD or No CKD using Random Forest.
    - Display model-generated classification scores.
    - View important features used by the model.
    - Run locally without an internet connection.
    """)

    st.warning(
        "Educational demonstration only. "
        "This model is not clinically validated and "
        "must not be used for medical diagnosis."
    )

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "Predict CKD":

    st.title("🩺 CKD Prediction")
    st.write(
        "Enter patient clinical measurements below."
    )

    with st.form("prediction_form"):

        st.subheader("Patient Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=45
            )

            bp = st.number_input(
                "Blood Pressure (mmHg)",
                min_value=40.0,
                max_value=250.0,
                value=80.0,
                step=1.0
            )

            sg = st.selectbox(
                "Urine Specific Gravity",
                [1.005, 1.010, 1.015, 1.020, 1.025],
                index=3
            )

            al = st.selectbox(
                "Urine Albumin Level (0-5)",
                [0, 1, 2, 3, 4, 5],
                index=0
            )

        with col2:

            bu = st.number_input(
                "Blood Urea (mg/dL)",
                min_value=1.0,
                max_value=400.0,
                value=40.0
            )

            sc = st.number_input(
                "Serum Creatinine (mg/dL)",
                min_value=0.1,
                max_value=80.0,
                value=1.2,
                step=0.1
            )

            hemo = st.number_input(
                "Hemoglobin (g/dL)",
                min_value=1.0,
                max_value=25.0,
                value=13.5,
                step=0.1
            )

            bgr = st.number_input(
                "Random Blood Glucose (mg/dL)",
                min_value=40.0,
                max_value=600.0,
                value=120.0
            )

        submitted = st.form_submit_button(
            "🔍 Predict CKD",
            use_container_width=True
        )

    if submitted:

        # Build input with exact training feature order
        input_data = pd.DataFrame([{
            "age": age,
            "bp": bp,
            "sg": sg,
            "al": al,
            "bu": bu,
            "sc": sc,
            "hemo": hemo,
            "bgr": bgr
        }])

        input_data = input_data[features]

        try:
            prediction = int(model.predict(input_data)[0])

            # Identify the probability of class 1 (CKD)
            class_labels = model.classes_
            ckd_index = list(class_labels).index(1)

            probability = model.predict_proba(
                input_data
            )[0][ckd_index]

            st.divider()
            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    "Prediction: Chronic Kidney Disease (CKD)"
                )

            else:

                st.success(
                    "Prediction: No CKD"
                )

            st.metric(
                "Model CKD Score",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

            st.caption(
                "The score represents the model's "
                "estimated class probability, not a "
                "clinically calibrated disease risk."
            )

            st.warning(
                "This result is for academic demonstration "
                "only. Consult a qualified healthcare "
                "professional for medical evaluation."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")

# -----------------------------
# MODEL INFORMATION PAGE
# -----------------------------
elif page == "Model Information":

    st.title("📊 Model Information")

    st.subheader("Machine Learning Model")

    st.write("""
    **Algorithm:** Random Forest Classifier

    **Dataset:** UCI Chronic Kidney Disease Dataset

    **Total Records:** 400

    **Selected Features:** 8

    **Training/Test Split:** 80:20

    **Preprocessing:** Median Imputation

    **Validation:** Stratified 5-Fold Cross-Validation
    """)

    st.subheader("Cross-Validation Results")

    cv_results = pd.DataFrame({
        "Fold": [
            "Fold 1", "Fold 2", "Fold 3",
            "Fold 4", "Fold 5"
        ],
        "Accuracy (%)": [
            100.00, 98.44, 96.88, 100.00, 100.00
        ]
    })

    st.dataframe(cv_results, use_container_width=True)

    st.metric("Mean Cross-Validation Accuracy", "99.06%")
    st.metric("Standard Deviation", "1.25%")

    st.bar_chart(
        cv_results.set_index("Fold")["Accuracy (%)"]
    )

    st.divider()

    st.subheader("Feature Importance")

    try:
        rf = model.named_steps["classifier"]

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": rf.feature_importances_
        }).sort_values(
            "Importance", ascending=False
        )

        st.bar_chart(
            importance.set_index("Feature")
        )

        st.dataframe(
            importance,
            use_container_width=True
        )

    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")

    st.info(
        "High feature importance means the model relies "
        "more heavily on that feature. It does not "
        "establish a causal relationship with CKD."
    )
