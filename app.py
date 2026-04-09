import streamlit as st
import pandas as pd
import pickle
import numpy as np
import eda
import streamlit.components.v1 as components

# Page Config
st.set_page_config(page_title="AI Health Risk Dashboard", layout="wide")

# Data & Model Loading
@st.cache_resource
def load_pipeline():
    try:
        with open('heart_attack_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_arm_rules():
    try:
        return pd.read_csv("heart_disease_rules.csv")
    except FileNotFoundError:
        return pd.DataFrame()
    
@st.cache_data
def load_raw_data():
    try:
        return pd.read_csv("heart_2022_no_nans.csv") 
    except FileNotFoundError:
        return pd.DataFrame()

pipeline = load_pipeline()
arm_df = load_arm_rules()
raw_df = load_raw_data()

age_map = {
    '18-24': 'Age 18 To 24',
    '25-29': 'Age 25 To 29',
    '30-34': 'Age 30 To 34',
    '35-39': 'Age 35 To 39',
    '40-44': 'Age 40 To 44',
    '45-49': 'Age 45 To 49',
    '50-54': 'Age 50 To 54',
    '55-59': 'Age 55 To 59',
    '60-64': 'Age 60 To 64',
    '65-69': 'Age 65 To 69',
    '70-74': 'Age 70 To 74',
    '75-79': 'Age 75 To 79',
    '80+': 'Age 80 or older'
}

def translate_antecedents(antecedents: str):
    mapping = {
        "BMI_Cat_Obese": "Obesity",
        "GeneralHealth_Poor": "Poor General Health",
        "HadAngina_Yes": "Angina",
        "HadDiabetes_Yes": "Diabetes",
        "DifficultyWalking_Yes": "Difficulty Walking",
        "HadStroke_Yes": "Previous Stroke",
        "Sex_Male": "Male",
        "SmokerStatus_Current smoker": "Current Smoker"
    }
    items = (
        antecedents.replace("frozenset(", "").replace(")", "")
        .replace("{", "").replace("}", "").replace("'", "")
        .split(",")
    )
    return " + ".join([mapping.get(i.strip(), i.strip()) for i in items])

def get_risk_level(lift):
    if lift >= 4: return "Very High"
    elif lift >= 2.5: return "High"
    else: return "Medium"

# UI Layout
st.title("🩺 AI Health Risk & Pattern Analysis")

if not pipeline or arm_df.empty:
    st.error("🚨 Missing required files ('heart_attack_pipeline.pkl' or 'heart_disease_rules.csv').")
    st.stop()

tab1, tab2 = st.tabs(["👤 Patient Analysis", "📊 Visual Exploratory"])

with tab1:
    st.subheader("👤 Predictive Assessment")

    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            sex = st.selectbox("Biological Sex", ["Female", "Male"])
            age = st.selectbox("Age Group", ['18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80+'])
            sleep_hours = st.number_input("Sleep Hour (Per Night)", 0.0, 24.0, 7.0, 0.1)
        with c2:
            weight = st.number_input("Weight (kg)", 30.0, 400.0, 70.0, 0.1)
            height = st.number_input("Height (cm)", 100.0, 300.0, 170.0, 0.1)
            gen_health = st.select_slider("General Health", ["Poor","Fair","Good","Very good","Excellent"], value="Good")
        with c3:
            smoker = st.selectbox("Smoking Status", ["Never Smoked","Former Smoker","Current Smoker"])
            e_cig = st.selectbox("E-Cigarette Use", ["Never Used","Use Them Some Days","Use Them Every Day"])
            
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        angina = "Yes" if c1.checkbox("Had Angina") else "No"
        stroke = "Yes" if c2.checkbox("Had Stroke") else "No"
        diabetes = "Yes" if c3.checkbox("Had Diabetes") else "No"
        walking = "Yes" if c4.checkbox("Difficulty Walking") else "No"

        submit = st.form_submit_button("🔍 Run Full Analysis", use_container_width=True)

    if submit:
        # --- 1. SVM Prediction ---
        svm_model = pipeline['svm_model']
        scaler = pipeline['scaler']
        pca = pipeline['pca_object']
        model_features = pipeline['feature_names']
        age_val = age_map[age]
        if e_cig == "Never Used":
            e_cig = "Not At All (Right Now)"
        if smoker == "Current Smoker":
            smoker = "Current smoker - Now Smokes Some Days"
        bmi = weight / ((height / 100) ** 2)

        input_df = pd.DataFrame([{
            'Sex': sex, 'AgeCategory': age_val, 'HadAngina': angina,
            'GeneralHealth': gen_health, 'SmokerStatus': smoker,
            'HadDiabetes': diabetes, 'BMI': bmi,
            'DifficultyWalking': walking, 'HadStroke': stroke,
            'SleepHours': sleep_hours, 'ECigaretteUsage': e_cig
        }])

        encoded = pd.get_dummies(input_df)
        full = pd.DataFrame(0, index=[0], columns=model_features)
        encoded_final = encoded.reindex(columns=model_features, fill_value=0)
        scaled = scaler.transform(encoded_final)
        scaled_df = pd.DataFrame(
            scaled,
            columns=encoded_final.columns
        )
        pca_data = pca.transform(scaled_df)

        score = svm_model.decision_function(pca_data)[0]

        # Save to session state for popup
        st.session_state.score = score
        st.session_state.bmi = round(bmi, 2)
        st.session_state.show_result = True
        
        # --- Three-Tier Risk Logic ---
        st.markdown("---")
        buffer = 0.5
        
        if st.session_state.show_result:
            @st.dialog("🧠 AI Health Risk Result")
            def result_popup():
                score = st.session_state.score
                bmi = st.session_state.bmi
                buffer = 0.5

                st.subheader(f"📏 Your BMI: {bmi}")

                if score > buffer:
                    st.error("⚠️ High Risk Profile")
                elif score < -buffer:
                    st.success("✅ Low Risk Profile")
                else:
                    st.warning("⚠️ Medium Risk Profile")

                st.write(f"Decision Score: **{score:.2f}**")
                st.write("Click x to close and scroll down to see associated rules")

            result_popup()
        # --- 2. Dynamic ARM Filtering ---
        st.subheader("🧠 Why this result? (Data-Driven Patterns)")
        user_features = []
        if bmi >= 30: user_features.append("BMI_Cat_Obese")
        if gen_health == "Poor": user_features.append("GeneralHealth_Poor")
        if angina == "Yes": user_features.append("HadAngina_Yes")
        if diabetes == "Yes": user_features.append("HadDiabetes_Yes")
        if stroke == "Yes": user_features.append("HadStroke_Yes")
        if sex == "Male": user_features.append("Sex_Male")
        if walking == "Yes": user_features.append("DifficultyWalking_Yes")
        if smoker == "Current smoker": user_features.append("SmokerStatus_Current smoker")
        
        def is_match(antecedent_str):
            ant_list = antecedent_str.replace("frozenset({", "").replace("})", "").replace("'", "").split(", ")
            return all(item.strip() in user_features for item in ant_list)

        matched_rules = arm_df[arm_df['antecedents'].apply(is_match)].copy()
        
        if not matched_rules.empty:
            # 1. First, generate the readable columns
            matched_rules['Pattern'] = matched_rules['antecedents'].apply(translate_antecedents)
            matched_rules['Risk Level'] = matched_rules['lift'].apply(get_risk_level)

            # 2. Define the sorting logic
            risk_order = {"Very High": 3, "High": 2, "Medium": 1}
            
            # 3. Create the sort key using the now-existing 'Risk Level' column
            matched_rules['sort_key'] = matched_rules['Risk Level'].map(risk_order)
            
            # 4. Sort descending (3 -> 2 -> 1)
            matched_rules = matched_rules.sort_values('sort_key', ascending=False)
            
            display_df = matched_rules[['Pattern', 'Risk Level']]
            st.info(f"Matched {len(display_df)} historical association rules based on your inputs:")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No specific high-risk patterns matched. The prediction is based on the SVM's overall feature interaction.")

        with st.expander("Technical Details"):
            st.write(f"SVM Decision Score: {score:.4f}")
            st.write(f"Detected Features: {', '.join(user_features) if user_features else 'None'}")

with tab2:
    if not raw_df.empty:
        eda.show_eda_page(raw_df)
    else:
        st.warning("⚠️ Raw dataset not found. Please ensure 'heart_2022_with_nans.csv' (or your data file) is in the directory.")