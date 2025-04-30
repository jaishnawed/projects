# inference.py

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

def load_models_and_preprocessor():
    """
    Load the trained models and preprocessor.
    """
    densenet128_model = tf.keras.models.load_model('./models/densenet128_model.h5')
    rf_model = joblib.load('./models/enhanced_rf_model.pkl')
    preprocessor = joblib.load('./models/preprocessor.pkl')
    return densenet128_model, rf_model, preprocessor

def preprocess_input_data(input_data, preprocessor):
    """
    Preprocess the input data using the saved preprocessor.
    """
    # Ensure the input data is in the correct format (e.g., DataFrame)
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_data = pd.DataFrame(input_data)
    
    # Drop the 'id' and 'diagnosis' columns if they exist (not needed for prediction)
    if 'id' in input_data.columns:
        input_data = input_data.drop(columns=['id'])
    if 'diagnosis' in input_data.columns:
        input_data = input_data.drop(columns=['diagnosis'])
    
    # Preprocess the data
    preprocessed_data = preprocessor.transform(input_data)
    return preprocessed_data

def predict_with_densenet128(model, preprocessed_data):
    """
    Make predictions using the DenseNet128 model.
    """
    predictions = model.predict(preprocessed_data)
    return (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

def predict_with_enhanced_rf(model, preprocessed_data):
    """
    Make predictions using the Enhanced Random Forest model.
    """
    predictions = model.predict(preprocessed_data)
    return predictions

def main():
    # Load the models and preprocessor
    densenet128_model, rf_model, preprocessor = load_models_and_preprocessor()

    # Example input data (replace this with actual new data)
    input_data = {
        'radius_mean': 17.99,
        'texture_mean': 10.38,
        'perimeter_mean': 122.8,
        'area_mean': 1001.0,
        'smoothness_mean': 0.1184,
        'compactness_mean': 0.2776,
        'concavity_mean': 0.3001,
        'concave points_mean': 0.1471,
        'symmetry_mean': 0.2419,
        'fractal_dimension_mean': 0.07871,
        'radius_se': 1.095,
        'texture_se': 0.9053,
        'perimeter_se': 8.589,
        'area_se': 153.4,
        'smoothness_se': 0.006399,
        'compactness_se': 0.04904,
        'concavity_se': 0.05373,
        'concave points_se': 0.01587,
        'symmetry_se': 0.03003,
        'fractal_dimension_se': 0.006193,
        'radius_worst': 25.38,
        'texture_worst': 17.33,
        'perimeter_worst': 184.6,
        'area_worst': 2019.0,
        'smoothness_worst': 0.1622,
        'compactness_worst': 0.6656,
        'concavity_worst': 0.7119,
        'concave points_worst': 0.2654,
        'symmetry_worst': 0.4601,
        'fractal_dimension_worst': 0.1189
    }

    # Preprocess the input data
    preprocessed_data = preprocess_input_data(input_data, preprocessor)

    # Make predictions using both models
    densenet128_prediction = predict_with_densenet128(densenet128_model, preprocessed_data)
    rf_prediction = predict_with_enhanced_rf(rf_model, preprocessed_data)

    # Print the predictions
    print("DenseNet128 Prediction:", "Malignant" if densenet128_prediction[0] == 1 else "Benign")
    print("Enhanced Random Forest Prediction:", "Malignant" if rf_prediction[0] == 1 else "Benign")

if __name__ == "__main__":
    main()