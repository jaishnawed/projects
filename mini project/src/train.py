# train.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def load_processed_data():
    """
    Load the preprocessed training and testing data.
    """
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    return X_train, X_test, y_train, y_test

def build_densenet128(input_shape):
    """
    Build the DenseNet128 model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_densenet128(X_train, y_train, X_test, y_test):
    """
    Train the DenseNet128 model and save it.
    """
    model = build_densenet128(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
    model.save('models/densenet128_model.h5')
    return model, history

def train_enhanced_rf(X_train, y_train):
    """
    Train the Enhanced Random Forest model and save it.
    """
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'models/enhanced_rf_model.pkl')
    return rf_model

def evaluate_model(model, X_test, y_test, model_type='densenet128'):
    """
    Evaluate the model on the test set and print metrics.
    """
    if model_type == 'densenet128':
        y_pred = (model.predict(X_test) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{model_type} Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

def main():
    # Load the preprocessed data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Train the DenseNet128 model
    print("Training DenseNet128 model...")
    densenet128_model, history = train_densenet128(X_train, y_train, X_test, y_test)
    evaluate_model(densenet128_model, X_test, y_test, model_type='densenet128')

    # Train the Enhanced Random Forest model
    print("Training Enhanced Random Forest model...")
    rf_model = train_enhanced_rf(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, model_type='enhanced_rf')

    print("Model training and evaluation completed.")

if __name__ == "__main__":
    main()