# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the dataset: handle missing values, encode categorical variables, and scale numerical features.
    """
    # Separate features and target
    X = data.drop(columns=['id', 'diagnosis'])  # Drop 'id' and 'diagnosis' columns
    y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Convert diagnosis to binary (1 for Malignant, 0 for Benign)

    # Define numerical and categorical features
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('scaler', StandardScaler())  # Standardize numerical features
    ])

    # Preprocessing for categorical features (if any)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply preprocessing
    X_preprocessed = preprocessor.fit_transform(X)

    # Save the preprocessor for later use
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    return X_preprocessed, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test):
    """
    Save the processed data to files.
    """
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)

def main():
    # Load the raw data
    data = load_data('data/raw/breast-cancer.csv')

    # Preprocess the data
    X, y = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save the processed data
    save_processed_data(X_train, X_test, y_train, y_test)

    print("Data preprocessing completed and saved.")

if __name__ == "__main__":
    main()