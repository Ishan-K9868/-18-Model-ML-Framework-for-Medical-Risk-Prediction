import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_NAMES = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'Glucose_HbA1c_Ratio']

st.set_page_config(
    page_title="🩺 Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f9f6;
    }
    .stButton > button {
        background-color: #387F39;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .stMetricValue {
        color: #387F39;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def load_model_package():
    try:
        model_pkg = joblib.load('final_diabetes_model.joblib')
        model = model_pkg.get('model')
        if model is None or not hasattr(model, 'predict'):
            st.error("Loaded package missing a valid model")
            return None
        scaler = joblib.load('scaler.pkl')
        model_pkg['feature_names'] = FEATURE_NAMES
        model_pkg['scaler'] = scaler
        return model_pkg
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None

def get_feature_importances(model_pkg):
    model = model_pkg['model']
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
    return None

def predict_diabetes(input_df, model_pkg, threshold=0.5):
    model = model_pkg['model']
    scaler = model_pkg['scaler']
    X = scaler.transform(input_df[FEATURE_NAMES])
    pos_idx = list(model.classes_).index(1)
    probs = model.predict_proba(X)[:, pos_idx]
    preds = (probs >= threshold).astype(int)
    return preds, probs

def main():
    st.title("🩺 Diabetes Prediction App")
    st.markdown("""
    Welcome to the **Diabetes Prediction App**.
    This tool uses a machine learning model trained on patient data to estimate the probability of diabetes.
    Please enter accurate medical information in the sidebar.
    """)

    pkg = load_model_package()
    if pkg is None:
        return

    feat_imp = get_feature_importances(pkg)

    st.subheader("🔑 Key Factors In Prediction")
    if feat_imp is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='Greens_d', ax=ax)
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        st.pyplot(fig)

    st.sidebar.header("🧾 Medical Information")
    user_input = {
        'age': st.sidebar.number_input("Age (years)", 0, 120, 33),
        'bmi': st.sidebar.number_input("BMI", 0.0, 100.0, 32.0),
        'blood_glucose_level': st.sidebar.number_input("Blood Glucose Level (mg/dL)", 50, 400, 120),
        'HbA1c_level': st.sidebar.number_input("HbA1c Level (%)", 3.0, 15.0, 5.7)
    }

    default_ratio = user_input['blood_glucose_level'] / max(user_input['HbA1c_level'], 0.1)
    user_input['Glucose_HbA1c_Ratio'] = st.sidebar.number_input(
        "Glucose to HbA1c Ratio", 0.0, 100.0, float(default_ratio), help="Auto-calculated but editable"
    )

    input_df = pd.DataFrame([user_input])

    if st.sidebar.button("🔍 Predict"):
        preds, probs = predict_diabetes(input_df, pkg, threshold=0.5)

        col1, col2 = st.columns(2)
        with col1:
            if preds[0] == 1:
                st.error("**Prediction: Positive for Diabetes ⚠️**")
            else:
                st.success("**Prediction: Negative for Diabetes ✅**")
            st.metric("Probability of Diabetes", f"{probs[0]:.2%}")

        with col2:
            fig, ax = plt.subplots(figsize=(6, 1.5))
            levels = [(0, 0.3, "Low"), (0.3, 0.6, "Moderate"), (0.6, 0.85, "High"), (0.85, 1.0, "Very High")]
            for s, e, label in levels:
                ax.barh(0, e - s, left=s, height=0.3, color=plt.cm.RdYlGn_r(s), alpha=0.7)
                ax.text((s + e) / 2, 0, label, ha='center', va='center', fontsize=9, fontweight='bold')
            ax.scatter(probs[0], 0, marker='v', s=300, color='black', zorder=5)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel('Diabetes Risk Probability')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            st.pyplot(fig)

        st.subheader("📋 Your Input Data")
        st.dataframe(input_df)

    st.sidebar.markdown("---")
    st.sidebar.header("About This App")
    st.sidebar.info("""
        This tool is part of the Diabetes Prediction Project.
        Developed using machine learning and Streamlit.
    """)

if __name__ == "__main__":
    main()
