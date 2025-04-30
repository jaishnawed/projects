# evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import tensorflow as tf

def load_models():
    """
    Load the trained models.
    """
    densenet128_model = tf.keras.models.load_model('models/densenet128_model.h5')
    rf_model = joblib.load('models/enhanced_rf_model.pkl')
    return densenet128_model, rf_model

def load_test_data():
    """
    Load the preprocessed test data.
    """
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    return X_test, y_test

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot and save the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'results/{model_name}_confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_prob, model_name):
    """
    Plot and save the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'results/{model_name}_roc_curve.png')
    plt.close()

def evaluate_densenet128(model, X_test, y_test):
    """
    Evaluate the DenseNet128 model.
    """
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    plot_confusion_matrix(y_test, y_pred, 'DenseNet128')
    plot_roc_curve(y_test, y_pred_prob, 'DenseNet128')
    print("DenseNet128 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

def evaluate_enhanced_rf(model, X_test, y_test):
    """
    Evaluate the Enhanced Random Forest model.
    """
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    plot_confusion_matrix(y_test, y_pred, 'EnhancedRF')
    plot_roc_curve(y_test, y_pred_prob, 'EnhancedRF')
    print("Enhanced Random Forest Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

def plot_feature_importance(model, X_test, feature_names):
    """
    Plot and save the feature importance for the Random Forest model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(X_test.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()

def main():
    # Load the trained models
    densenet128_model, rf_model = load_models()

    # Load the test data
    X_test, y_test = load_test_data()

    # Evaluate the DenseNet128 model
    print("Evaluating DenseNet128 model...")
    evaluate_densenet128(densenet128_model, X_test, y_test)

    # Evaluate the Enhanced Random Forest model
    print("Evaluating Enhanced Random Forest model...")
    evaluate_enhanced_rf(rf_model, X_test, y_test)

    # Plot feature importance for the Random Forest model
    # Assuming feature names are available (replace with actual feature names if needed)
    feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    plot_feature_importance(rf_model, X_test, feature_names)

    print("Model evaluation completed. Results saved in the 'results/' directory.")

if __name__ == "__main__":
    main()