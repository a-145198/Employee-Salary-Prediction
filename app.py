import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np

st.title("Model Performance Comparison and Prediction")

# Load saved models and their accuracies
model_dir = "saved_models"
models_data = {
    "Logistic Regression": {"accuracy": None, "model": None},
    "Random Forest": {"accuracy": None, "model": None},
    "K-Nearest Neighbors": {"accuracy": None, "model": None},
    "Support Vector Machine": {"accuracy": None, "model": None},
    "Gradient Boosting": {"accuracy": None, "model": None},
    "Tuned Gradient Boosting": {"accuracy": None, "model": None}
}

# Define a dictionary to map model names to filenames
model_filenames = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "K-Nearest Neighbors": "k-nearest_neighbors_model.pkl",
    "Support Vector Machine": "support_vector_machine_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Tuned Gradient Boosting": "tuned_gradient_boosting_model.pkl"
}

# Load accuracies from a dictionary (you would ideally save this dictionary)
# For now, we'll hardcode or load from a file if saved previously
# Assuming 'results' and 'test_accuracy_gb' from the notebook are available or saved
# If not saved, you would need to re-run the model training or load from a saved file
# For this example, let's use the accuracies from the notebook output
accuracies = {
    "Logistic Regression": 0.8511,
    "Random Forest": 0.8514,
    "K-Nearest Neighbors": 0.8220,
    "Support Vector Machine": 0.8520,
    "Gradient Boosting": 0.8689,
    "Tuned Gradient Boosting": 0.8748
}

for name, filename in model_filenames.items():
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        models_data[name]["model"] = joblib.load(filepath)
        models_data[name]["accuracy"] = accuracies.get(name) # Get accuracy from the dictionary
    else:
        st.warning(f"Model file not found: {filepath}")


# Create a bar chart of the accuracies
model_names = list(models_data.keys())
accuracy_values = [data["accuracy"] for data in models_data.values() if data["accuracy"] is not None]
model_names_filtered = [name for name, data in models_data.items() if data["accuracy"] is not None]

if accuracy_values:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names_filtered, y=accuracy_values, palette='viridis', hue=model_names_filtered, legend=False)
    plt.ylim(0, 1)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)
else:
    st.warning("No model accuracies available to plot.")

st.write("---")
st.subheader("Make a Prediction (using Tuned Gradient Boosting)")

# Load the scaler and training columns
scaler_filepath = os.path.join(model_dir, "scaler.pkl")
training_columns_filepath = os.path.join(model_dir, "training_columns.pkl")

scaler = None
training_columns = None

if os.path.exists(scaler_filepath):
    scaler = joblib.load(scaler_filepath)
else:
    st.error(f"Scaler file not found: {scaler_filepath}")

if os.path.exists(training_columns_filepath):
    training_columns = joblib.load(training_columns_filepath)
else:
    st.error(f"Training columns file not found: {training_columns_filepath}")


tuned_gb_model = models_data.get("Tuned Gradient Boosting", {}).get("model")

if tuned_gb_model and scaler and training_columns:
    st.write("Tuned Gradient Boosting Model is loaded and ready for predictions.")

    # Add input widgets for features
    st.write("Enter the features for prediction:")

    # Define your features and their types (update this based on your actual features)
    # This is crucial for correctly handling numerical and categorical inputs
    feature_info = {
        'age': {'type': 'slider', 'range': (18, 90), 'default': 30},
        'workclass': {'type': 'selectbox', 'options': ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked']},
        'fnlwgt': {'type': 'number_input', 'range': (0, 1500000), 'default': 200000}, # Example range, adjust as needed
        'education': {'type': 'selectbox', 'options': ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', 'Assoc-acdm', '11th', '10th', '7th-8th', 'Prof-school', '9th', '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool']},
        'educational-num': {'type': 'number_input', 'range': (1, 16), 'default': 9}, # Example range, adjust as needed
        'marital-status': {'type': 'selectbox', 'options': ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']},
        'occupation': {'type': 'selectbox', 'options': ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']},
        'relationship': {'type': 'selectbox', 'options': ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']},
        'race': {'type': 'selectbox', 'options': ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']},
        'gender': {'type': 'selectbox', 'options': ['Male', 'Female']},
        'capital-gain': {'type': 'number_input', 'range': (0, 100000), 'default': 0}, # Example range, adjust as needed
        'capital-loss': {'type': 'number_input', 'range': (0, 5000), 'default': 0},   # Example range, adjust as needed
        'hours-per-week': {'type': 'slider', 'range': (1, 99), 'default': 40},
        'native-country': {'type': 'selectbox', 'options': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']} # Add all unique values from your data
    }

    input_data = {}
    for feature, details in feature_info.items():
        if details['type'] == 'slider':
            input_data[feature] = st.slider(feature.replace('_', ' ').title(), details['range'][0], details['range'][1], details['default'])
        elif details['type'] == 'selectbox':
            input_data[feature] = st.selectbox(feature.replace('_', ' ').title(), details['options'])
        elif details['type'] == 'number_input':
             input_data[feature] = st.number_input(feature.replace('_', ' ').title(), min_value=details['range'][0], max_value=details['range'][1], value=details['default'])


    if st.button("Predict Income"):
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])

            # Apply one-hot encoding to categorical features
            categorical_cols = [f for f, d in feature_info.items() if d['type'] == 'selectbox']
            input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

            # Ensure all columns from training are present (add missing columns with 0)
            # This handles cases where a category might not be present in the current input
            for col in training_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Reorder columns to match training data
            input_df = input_df[training_columns]

            # Scale numeric features
            # Identify numeric columns based on the feature_info and training columns
            numeric_cols_train = [col for col in training_columns if col in [f for f, d in feature_info.items() if d['type'] in ['slider', 'number_input']]]
            input_df[numeric_cols_train] = scaler.transform(input_df[numeric_cols_train])


            # Make prediction
            prediction = tuned_gb_model.predict(input_df)
            predicted_income = '<=50K' if prediction[0] == 0 else '>50K'
            st.success(f"Predicted Income: {predicted_income}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.info("Tuned Gradient Boosting Model, scaler, or training columns not loaded. Cannot make predictions.")
