import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the final model and related artifacts
@st.cache_resource
def load_model():
    try:
        model_package = joblib.load('final_diabetes_model.joblib')
        if not hasattr(model_package['model'], 'predict'):
            st.error("Loaded model missing 'predict' method")
            return None
        print(f"Loaded model: {model_package['model_name']}")
        print(f"Features: {model_package['feature_names']}")
        return model_package
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Prediction logic
def predict_diabetes(input_array, model_package, threshold=0.5):
    model = model_package['model']
    scaler = model_package.get('scaler')
    model_name = model_package['model_name']

    # Scale
    if scaler is not None:
        input_array = scaler.transform(input_array)

    # Get probabilities
    if 'Neural Network' in model_name:
        probs = model.predict(input_array).flatten()
    else:
        prob_array = model.predict_proba(input_array)
        classes = model.classes_
        pos_index = list(classes).index(1) if 1 in classes else None
        if pos_index is None:
            st.error("Positive class '1' not found in model.classes_")
            return None, None
        probs = prob_array[:, pos_index]

    preds = (probs >= threshold).astype(int)
    return preds, probs

# Main app
def main():
    st.title("Diabetes Prediction App")
    
    # Add model and project description
    st.markdown("""
        ### About This Model
        This application uses a machine learning model to predict the likelihood of diabetes based on various health metrics.
        The model has been trained on a Diabetes Database and optimized for accuracy and interpretability.
        
        ### Project Description
        This project aims to provide a user-friendly tool for preliminary diabetes risk assessment.
        By entering your medical information, you can get an instant prediction of your diabetes risk level.
        **Note:** This tool is for educational purposes only and should not replace professional medical advice.
    """)

    st.sidebar.header("Enter Medical Information")

    # Basic inputs
    user_input = {
        'Pregnancies': st.sidebar.number_input("Pregnancies", 0, 17, 0),
        'Glucose': st.sidebar.number_input("Glucose (mg/dL)", 0, 200, 120),
        'BloodPressure': st.sidebar.number_input("Blood Pressure (mm Hg)", 0, 122, 70),
        'SkinThickness': st.sidebar.number_input("Skin Thickness (mm)", 0, 100, 20),
        'Insulin': st.sidebar.number_input("Insulin (mu U/ml)", 0, 846, 79),
        'BMI': st.sidebar.number_input("BMI", 0.0, 67.0, 32.0),
        'DiabetesPedigreeFunction': st.sidebar.number_input("Diabetes Pedigree Function", 0.078, 2.42, 0.47),
        'Age': st.sidebar.number_input("Age", 21, 81, 33),
    }

    # Derived categories
    bmi_val = user_input['BMI']
    glucose_val = user_input['Glucose']
    age_val = user_input['Age']

    bmi_cat = "Underweight" if bmi_val < 18.5 else "Normal" if bmi_val < 25 else "Overweight" if bmi_val < 30 else "Obese"
    gluc_cat = "Low" if glucose_val < 70 else "Normal" if glucose_val < 140 else "High"
    age_cat = "Young" if age_val < 30 else "Middle-aged" if age_val < 50 else "Elderly"

    st.sidebar.subheader("Advanced Features")
    user_input['BMI_Category'] = st.sidebar.selectbox("BMI Category", ["Underweight","Normal","Overweight","Obese"], index=["Underweight","Normal","Overweight","Obese"].index(bmi_cat))
    user_input['Glucose_Level'] = st.sidebar.selectbox("Glucose Level", ["Low","Normal","High"], index=["Low","Normal","High"].index(gluc_cat))
    user_input['Age_Group'] = st.sidebar.selectbox("Age Group", ["Young","Middle-aged","Elderly"], index=["Young","Middle-aged","Elderly"].index(age_cat))
    insulin_safe = max(user_input['Insulin'], 1)
    user_input['Glucose_to_Insulin_Ratio'] = glucose_val / insulin_safe
    est_hba1c = (glucose_val + 46.7) / 28.7
    user_input['HbA1c_Level'] = st.sidebar.number_input("HbA1c Level", 0.0, 20.0, float(est_hba1c))
    user_input['Glucose_HbA1c_Ratio'] = st.sidebar.number_input(
        "Glucose to HbA1c Ratio", 0.0, 100.0, float(glucose_val/max(user_input['HbA1c_Level'],0.1))
    )

    # Build DataFrame exactly matching model's expected features
    model_package = load_model()
    if model_package is None:
        return
    feature_names = model_package['feature_names']

    aligned = {}
    for feat in feature_names:
        # direct key
        if feat in user_input:
            aligned[feat] = user_input[feat]
        else:
            # case-insensitive match
            match = next((v for k, v in user_input.items() if k.lower() == feat.lower()), 0.0)
            aligned[feat] = match

    input_df = pd.DataFrame([aligned])

    if st.sidebar.button("Predict"):
        preds, probs = predict_diabetes(input_df.values, model_package, threshold=0.2)
        if preds is None:
            return

        col1, col2 = st.columns(2)
        with col1:
            if preds[0] == 1:
                st.error("**Positive for Diabetes** ⚠️")
            else:
                st.success("**Negative for Diabetes** ✅")
            st.metric("Probability of Diabetes", f"{probs[0]:.2%}")

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            levels = [(0,0.2,"Low"),(0.2,0.5,"Moderate"),(0.5,0.7,"High"),(0.7,1.0,"Very High")]
            for s,e,l in levels:
                ax.barh(0, e-s, left=s, height=0.3, color=plt.cm.RdYlGn_r(s*0.8+0.2), alpha=0.8)
                ax.text((s+e)/2, 0, l, ha='center', va='center', fontweight='bold')
            ax.scatter(probs[0], 0, marker='v', s=400, zorder=5)
            ax.set_xlim(0,1)
            ax.get_yaxis().set_visible(False)
            ax.set_xlabel('Probability of Diabetes')
            for spine in ['top','left','right']:
                ax.spines[spine].set_visible(False)
            st.pyplot(fig)

        st.header("Your Input Data")
        st.dataframe(input_df)

    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info(
        """
        This application is part of the comprehensive Diabetes Prediction Project
        that explores 18 different machine learning models for diabetes prediction.

        [View Project on GitHub](https://github.com/)
        """
    )


if __name__ == "__main__":
    main()
