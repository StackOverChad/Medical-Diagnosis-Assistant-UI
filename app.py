import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import json
import os
import joblib # For scikit-learn models and related artifacts
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="💉", # Using a syringe emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress Keras/TF warnings on startup for cleaner UI
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# --- Configuration for ONLY the listed models ---
# IMPORTANT: Ensure these filenames match your downloaded files exactly.
# Rename your class names files (if no extension) to end with .json (e.g., 'my_class_names.json').
# Place ALL these files in the same directory as this app.py script.

MODEL_CONFIGS = {
    "Symptom-Based": {
        "model_path": "symptom_prediction_model.joblib",
        "label_encoder_path": "symptom_label_encoder.joblib",
        "feature_columns_path": "symptom_feature_columns.joblib",
        "description": "Predicts a disease from a list of symptoms.",
        "type": "tabular",
        "input_example": {"Symptoms": ["headache", "high fever", "vomiting"]}, # Placeholder for symptom input
    },
    "Heart Disease": {
        "model_path": "heart_failure_prediction_model.joblib",
        "feature_columns_path": "heart_failure_feature_columns.joblib",
        "class_names": ['No Heart Disease', 'Heart Disease'],
        "description": "Predicts heart disease from health metrics.",
        "type": "tabular",
        "input_example": {
            "Age": 55, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 130, "Cholesterol": 240,
            "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 150, "ExerciseAngina": "N",
            "Oldpeak": 1.5, "ST_Slope": "Flat"
        },
        "categorical_features": ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    },
    "Diabetes": {
        "model_path": "diabetes_prediction_model.joblib",
        "feature_columns_path": "diabetes_feature_columns.joblib",
        "class_names": ['No Diabetes', 'Diabetes'],
        "description": "Predicts diabetes from diagnostic measurements.",
        "type": "tabular",
        "input_example": {
            'Pregnancies': 2, 'Glucose': 120, 'BloodPressure': 70, 'SkinThickness': 30,
            'Insulin': 100, 'BMI': 30.5, 'DiabetesPedigreeFunction': 0.5, 'Age': 35
        },
        "zero_to_nan_cols": ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    },
    "CKD (Kidney Disease)": {
        "model_path": "ckd_prediction_model.joblib",
        "feature_columns_path": "ckd_feature_columns.joblib",
        "label_encoder_path": "ckd_label_encoder.joblib", # This one uses a LabelEncoder
        "description": "Predicts chronic kidney disease.",
        "type": "tabular",
        "input_example": {
            'age': 48, 'bp': 80, 'sg': 1.020, 'al': 1, 'su': 0, 'rbc': 'normal',
            'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent', 'bgr': 121,
            'bu': 36, 'sc': 1.2, 'sod': 135, 'pot': 4.7, 'hemo': 15.4, 'pcv': 44,
            'wc': 7800, 'rc': 5.2, 'htn': 'yes', 'dm': 'yes', 'cad': 'no',
            'appet': 'good', 'pe': 'no', 'ane': 'no'
        },
        "categorical_features_order": ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'],
        "numerical_features_order": ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    },
    "Stroke": {
        "model_path": "stroke_prediction_model.joblib",
        "feature_columns_path": "stroke_feature_columns.joblib",
        "class_names": ['No Stroke', 'Stroke'],
        "description": "Predicts stroke risk.",
        "type": "tabular",
        "input_example": {
            "gender": "Male", "age": 67.0, "hypertension": 0, "heart_disease": 1,
            "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
            "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked"
        },
        "categorical_features": ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]
    },
    "Chest X-Ray (Pneumonia)": {
        "model_path": "chest_xray_pneumonia_best_model.keras",
        "class_names_path": "chest_xray_pneumonia_class_names.json", # Ensure this is renamed to .json
        "description": "Detects pneumonia from chest X-ray images.",
        "type": "image",
        "problem_type": "binary",
        "target_size": (224, 224)
    },
    "COVID-19 Radiography": {
        "model_path": "covid19_radiography_best_model.keras",
        "class_names_path": "covid19_radiography_class_names.json", # Ensure this is renamed to .json
        "description": "Classifies chest X-rays for COVID-19, Normal, or Viral Pneumonia.",
        "type": "image",
        "problem_type": "categorical",
        "target_size": (224, 224)
    },
    "Breast Histopathology": {
        "model_path": "breast_histopathology_best_model.keras",
        "class_names_path": "breast_histopathology_class_names.json",
        "description": "Classifies breast histopathology images (benign/malignant patches).",
        "type": "image",
        "problem_type": "binary",
        "target_size": (96, 96) # Smaller size as it's for patches
    },
}

# --- Load all models and their associated artifacts ---
@st.cache_resource # Use Streamlit's cache to load models once
def load_all_models():
    loaded_models = {}
    for key, config in MODEL_CONFIGS.items():
        try:
            if config["type"] == "tabular":
                model = joblib.load(config["model_path"])
                feature_columns = joblib.load(config["feature_columns_path"])
                label_encoder = None
                class_names = config.get("class_names")

                if "label_encoder_path" in config:
                    label_encoder = joblib.load(config["label_encoder_path"])
                    class_names = label_encoder.classes_.tolist()

                loaded_models[key] = {
                    "model": model,
                    "feature_columns": feature_columns,
                    "label_encoder": label_encoder,
                    "class_names": class_names,
                    "type": "tabular",
                    "description": config["description"],
                    "input_example": config.get("input_example", {}),
                    "categorical_features": config.get("categorical_features", []),
                    "numerical_features_order": config.get("numerical_features_order", []),
                    "categorical_features_order": config.get("categorical_features_order", []),
                    "zero_to_nan_cols": config.get("zero_to_nan_cols", [])
                }
            elif config["type"] == "image":
                model = load_model(config["model_path"])
                with open(config["class_names_path"], 'r') as f:
                    class_names = json.load(f)
                loaded_models[key] = {
                    "model": model,
                    "class_names": class_names,
                    "type": "image",
                    "problem_type": config["problem_type"],
                    "target_size": config["target_size"],
                    "description": config["description"]
                }
            st.sidebar.success(f"Loaded {key} model.")
        except FileNotFoundError as e:
            st.sidebar.error(f"Missing file for {key} model: {e}. Please ensure all model files are in the same directory.")
            # Store a placeholder to indicate failure, but keep description available
            loaded_models[key] = {"model": None, "description": config["description"], "type": config["type"], "load_error": str(e)}
        except Exception as e:
            st.sidebar.error(f"Failed to load {key} model: {e}. Check file integrity or format.")
            # Store a placeholder
            loaded_models[key] = {"model": None, "description": config["description"], "type": config["type"], "load_error": str(e)}
    return loaded_models

MODELS = load_all_models()

# --- Prediction Functions (for tabular models) ---
def predict_symptom_based(model_pipeline, symptom_encoder, user_symptoms_input_list, all_symptom_columns_expected):
    input_vector = pd.DataFrame(0, index=[0], columns=all_symptom_columns_expected)
    for symptom in user_symptoms_input_list:
        formatted_symptom = symptom.strip().lower().replace(" ", "_")
        if formatted_symptom in all_symptom_columns_expected:
            input_vector[formatted_symptom] = 1
    
    # Ensure input_vector is exactly like X_train
    input_vector = input_vector[all_symptom_columns_expected]

    prediction_encoded = model_pipeline.predict(input_vector)[0]
    predicted_disease = symptom_encoder.inverse_transform([prediction_encoded])[0]

    probabilities = model_pipeline.predict_proba(input_vector)[0]
    top_indices = probabilities.argsort()[-5:][::-1]
    top_diseases_with_probs = [(symptom_encoder.inverse_transform([i])[0], probabilities[i]*100) for i in top_indices]
    return predicted_disease, top_diseases_with_probs

def predict_general_tabular(model_pipeline, processed_input_df, class_names, label_encoder=None):
    probabilities = model_pipeline.predict_proba(processed_input_df)[0]
    predicted_label_idx = np.argmax(probabilities)
    predicted_label_prob = probabilities[predicted_label_idx] * 100

    if label_encoder:
        predicted_class_name = label_encoder.inverse_transform([predicted_label_idx])[0]
        all_class_names = label_encoder.classes_.tolist()
    else:
        predicted_class_name = class_names[predicted_label_idx]
        all_class_names = class_names
    
    prob_df = pd.DataFrame({'Disease/Status': all_class_names, 'Probability': probabilities * 100})
    prob_df = prob_df.sort_values(by='Probability', ascending=False)
    
    return predicted_class_name, predicted_label_prob, prob_df

# --- UI Layout ---
st.title("💉 Medical Diagnosis Assistant")
st.markdown("""
Welcome to the Medical Diagnosis Assistant! Use this tool to get AI-powered predictions
based on symptoms or medical images.

**Disclaimer:** This tool is for educational purposes only and should **NOT** be used for actual medical diagnosis or treatment. Always consult a qualified medical professional.
""")
st.markdown("---")

# --- Sidebar for Model Selection ---
available_model_names = [name for name, info in MODELS.items() if info["model"] is not None]

if not available_model_names:
    st.sidebar.error("No models were loaded successfully. Please check the error messages above and ensure all files are correct.")
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Choose a medical model:",
    available_model_names
)

st.sidebar.markdown("---")
st.sidebar.info("Ensure all model files are in the same directory as this script and named correctly.")

# --- Dynamic Content Based on Selected Model ---
if selected_model_name and selected_model_name in MODELS:
    current_model_info = MODELS[selected_model_name]

    if current_model_info["model"] is None:
        st.error(f"Error: The model for {selected_model_name} could not be loaded. Please check the logs and the file `{current_model_info.get('model_path', 'unknown_path')}`.")
        st.stop()

    st.header(f"🩺 {current_model_info['description']}")

    # --- Tabular Model Input/Prediction ---
    if current_model_info["type"] == "tabular":
        st.subheader("Enter Patient Data")

        input_data_dict = {}
        example_data = current_model_info["input_example"]

        if selected_model_name == "Symptom-Based":
            symptoms_text = st.text_area(
                "Enter symptoms you are experiencing (comma-separated, e.g., 'headache, high fever, vomiting'):",
                value="headache, high fever"
            )
            user_symptoms_list = [s.strip() for s in symptoms_text.split(',') if s.strip()]
            input_data_dict["Symptoms"] = user_symptoms_list
            
            st.info("Recognized symptoms are based on the training data. Common ones include: abdominal_pain, high_fever, fatigue, vomiting, skin_rash, chills, joint_pain, headache, muscle_pain. Please try to match these formats.")

        else:
            cols = st.columns(2)
            for i, (feature, ex_value) in enumerate(example_data.items()):
                with cols[i % 2]:
                    if isinstance(ex_value, str) or feature in current_model_info.get('categorical_features', []):
                        input_data_dict[feature] = st.text_input(f"{feature}:", value=str(ex_value))
                    elif isinstance(ex_value, int) or feature in current_model_info.get('numerical_features_order', []):
                        input_data_dict[feature] = st.number_input(f"{feature}:", value=float(ex_value), format="%f", step=1.0)
                    elif isinstance(ex_value, float):
                        input_data_dict[feature] = st.number_input(f"{feature}:", value=ex_value, format="%.2f")
                    else:
                        input_data_dict[feature] = st.text_input(f"{feature}: (unsupported type)", value=str(ex_value))

        if st.button("Predict"):
            if not current_model_info["model"]:
                st.error("Model not loaded. Please check the sidebar for errors.")
            else:
                try:
                    if selected_model_name == "Symptom-Based":
                        if not input_data_dict["Symptoms"]:
                            st.warning("Please enter at least one symptom.")
                        else:
                            predicted_disease, top_diseases_with_probs = predict_symptom_based(
                                current_model_info["model"],
                                current_model_info["label_encoder"],
                                input_data_dict["Symptoms"],
                                current_model_info["feature_columns"]
                            )
                            st.success(f"**Most Probable Diagnosis: {predicted_disease}**")
                            st.subheader("Top Probable Diagnoses:")
                            for disease, prob in top_diseases_with_probs:
                                st.write(f"- {disease}: {prob:.2f}%")
                    else:
                        user_input_for_df = {}
                        for k, v in input_data_dict.items():
                            if v is None or v == '':
                                user_input_for_df[k] = np.nan
                            else:
                                try:
                                    if k in current_model_info.get('numerical_features_order', []) or \
                                       (k in current_model_info.get('input_example', {}) and isinstance(current_model_info.get('input_example', {})[k], (int, float))):
                                        user_input_for_df[k] = float(v)
                                    else:
                                        user_input_for_df[k] = v
                                except ValueError:
                                    user_input_for_df[k] = v

                        processed_input_df = pd.DataFrame([user_input_for_df])

                        if selected_model_name == "Heart Disease":
                            categorical_cols = current_model_info["categorical_features"]
                            processed_input_df = pd.get_dummies(processed_input_df, columns=categorical_cols, drop_first=True)
                            
                        elif selected_model_name == "Diabetes":
                            for col in current_model_info["zero_to_nan_cols"]:
                                if col in processed_input_df.columns:
                                    processed_input_df[col] = processed_input_df[col].replace(0.0, np.nan)

                        elif selected_model_name == "CKD (Kidney Disease)":
                            categorical_features_order = current_model_info["categorical_features_order"]
                            numerical_features_order = current_model_info["numerical_features_order"]
                            
                            for col in numerical_features_order:
                                if col in processed_input_df.columns:
                                    processed_input_df[col] = pd.to_numeric(processed_input_df[col], errors='coerce') 

                            temp_processed_df = processed_input_df.copy()
                            for cat_col in categorical_features_order:
                                if cat_col in temp_processed_df.columns:
                                    temp_processed_df = pd.get_dummies(temp_processed_df, columns=[cat_col], prefix=cat_col, drop_first=True)
                            processed_input_df = temp_processed_df.copy()
                            
                        elif selected_model_name == "Stroke":
                            categorical_cols = current_model_info["categorical_features"]
                            for col in ['hypertension', 'heart_disease']:
                                if col in processed_input_df.columns:
                                    processed_input_df[col] = processed_input_df[col].astype(object)
                            processed_input_df = pd.get_dummies(processed_input_df, columns=categorical_cols, drop_first=True)

                        expected_features = current_model_info["feature_columns"]
                        final_input_for_prediction = processed_input_df.reindex(columns=expected_features, fill_value=0)

                        predicted_class_name, predicted_label_prob, prob_df = predict_general_tabular(
                            current_model_info["model"],
                            final_input_for_prediction,
                            current_model_info["class_names"],
                            current_model_info.get("label_encoder")
                        )

                        st.success(f"**Predicted: {predicted_class_name}**")
                        st.write(f"Confidence: {predicted_label_prob:.2f}%")
                        st.subheader("All Probabilities:")
                        st.dataframe(prob_df)

                except Exception as e:
                    st.error(f"An unexpected error occurred during prediction for {selected_model_name}: {e}")
                    st.warning("Please ensure all input values are correct and match the expected format (e.g., 'Male' vs 'M' for gender, 'Yes' vs 'Y' for binary features).")
                    st.info("If the error persists, check the terminal where Streamlit is running for detailed traceback.")


    # --- Image Model Input/Prediction ---
    elif current_model_info["type"] == "image":
        st.subheader("Upload Medical Image")
        uploaded_file = st.file_uploader(f"Choose a {selected_model_name} image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

            img_width, img_height = current_model_info["target_size"]
            img = image.load_img(uploaded_file, target_size=(img_width, img_height))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            st.subheader("Prediction:")
            with st.spinner('Analyzing image...'):
                predictions = current_model_info["model"].predict(img_array)

            class_names = current_model_info["class_names"]
            problem_type = current_model_info["problem_type"]

            if problem_type == 'binary':
                prediction_prob = predictions[0][0]
                predicted_class_idx = 1 if prediction_prob >= 0.5 else 0
                predicted_class = class_names[predicted_class_idx]
                confidence = prediction_prob if predicted_class_idx == 1 else (1 - prediction_prob)

                st.success(f"**Predicted: {predicted_class}**")
                st.write(f"Confidence: {confidence*100:.2f}%")

                st.subheader("All Probabilities:")
                prob_df = pd.DataFrame({
                    'Class': [class_names[0], class_names[1]],
                    'Probability': [(1 - prediction_prob) * 100, prediction_prob * 100] # Assuming class_names[0] is negative, class_names[1] is positive
                })
                prob_df = prob_df.sort_values(by='Probability', ascending=False)
                st.dataframe(prob_df)


            elif problem_type == 'categorical':
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = class_names[predicted_class_idx]
                confidence = predictions[0][predicted_class_idx] * 100

                st.success(f"**Predicted: {predicted_class}** (Confidence: {confidence:.2f}%)")

                st.subheader("Top Probable Diagnoses:")
                top_3_indices = predictions[0].argsort()[-3:][::-1]
                for idx in top_3_indices:
                    st.write(f"- {class_names[idx]}: {predictions[0][idx]*100:.2f}%")

            elif problem_type == 'raw': # Multi-label classification
                predicted_labels_binary = (predictions[0] > 0.5).astype(int)
                positive_predictions = [class_names[i] for i, val in enumerate(predicted_labels_binary) if val == 1]

                if positive_predictions:
                    st.success(f"**Predicted Positive for: {', '.join(positive_predictions)}**")
                else:
                    st.info("Predicted Negative for all known conditions.")

                st.subheader("Probabilities for All Labels:")
                prob_df = pd.DataFrame({
                    'Label': class_names,
                    'Probability': predictions[0] * 100,
                    'Predicted (Threshold 0.5)': predicted_labels_binary
                })
                st.dataframe(prob_df.sort_values(by='Probability', ascending=False))

        else:
            st.info("Please upload an image to get a diagnosis.")