import os
import pickle
import pandas as pd
import streamlit as st

st.title('Diabetes Prediction App')

MODELS_FOLDER = 'models'
MODEL_FILES = {
    "random_forest": "random_forest.pkl",
    "decision_tree": "decision_tree.pkl",
    "knn": "knn.pkl",
}

@st.cache_resource
def load_models():
    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():
        # Check if the model pkl file is provided in the corresponding folder by the corresponding filename
        model_path = os.path.join(MODELS_FOLDER, filename)
        # If exists, load the model using pickle
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                loaded_models[model_name] = pickle.load(f)

    # Return the list of all the models that were available in the folder
    return loaded_models

# Load saved models
models = load_models()
# Get the list of the available models
available_model_names = list(models.keys())

# If no models could have been loaded,
# Display the error and the description of how to fix the issue.
# Stop the streamlit application as well.
if not models:
    st.error("No models found")
    st.info(
        "Please train and save the models before running this app.\n"
        "At least one of this files should exist in the application directory:\n"
        "- models/random_forest.pkl\n"
        "- models/decision_tree.pkl\n"
        "- models/knn.pkl"
    )
    st.stop()

# Form definition
with st.form(key='form'):
    # Inputs
    age = st.number_input('Your age', min_value=1, max_value=90)
    bmi = st.number_input('Your bmi', min_value=10.0, max_value=100.0)
    HbA1c_level = st.number_input('Your HbA1c level', min_value=3.5, max_value=9.0)
    blood_glucose_level = st.number_input('Your blood glucose level', min_value=80.0, max_value=300.0)
    gender = st.selectbox(
        'Your gender',
        ('Male', 'Female')
    )
    hypertension_inp = st.selectbox(
        'Do you have a hypertension?',
        ('Yes', 'No')
    )
    heart_disease_inp = st.selectbox(
        'Do you have or have you had a heart disease?',
        ('Yes', 'No')
    )
    smoking_history_inp = st.selectbox(
        'What is your smoking history?',
        ('Never smoked', 'Smoked in the past', 'I am a smoker')
    )
    selected_model = st.selectbox("Choose a model", available_model_names)
    # Submit button
    submitted = st.form_submit_button('Predict if i have diabetes')

# Action to perform on the form submit
if submitted:
    # Clean up the data
    smoking_history = {
        "Never smoked": 'not_smoker',
        "Smoked in the past": 'past_smoker',
        "I am a smoker": 'smoker'
    }[smoking_history_inp]
    hypertension = 1 if hypertension_inp == 'Yes' else 0
    heart_disease = 1 if heart_disease_inp == 'Yes' else 0

    # Assemble the X dataset
    input_df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level,
    }])

    # Get the selected model
    model = models[selected_model]
    # Predict diabetes
    prediction = model.predict(input_df)

    # Display the prediction
    if prediction[0] == 1:
        st.error("The model predicts diabetes")
    else:
        st.success("The model predicts no diabetes")
