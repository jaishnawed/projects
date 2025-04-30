from flask import Flask, request, jsonify, send_from_directory
import sqlite3
import os
import pandas as pd
from pathlib import Path
import joblib
import tensorflow as tf
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

BASE_DIR = Path(__file__).parent

# Load the models
rf_model = joblib.load(BASE_DIR / './models/enhanced_rf_model.pkl')
densenet_model = tf.keras.models.load_model(BASE_DIR / './models/densenet128_model.h5')

# Database initialization
def init_db():
    db_path = BASE_DIR / 'users.db'
    if not db_path.exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                         ('wizards', 'ishukant'))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        conn.close()

init_db()

# Ensure Excel file exists
file_path = BASE_DIR / 'test_results.xlsx'
columns = ['Timestamp', 'Username'] + [f'feature_{i}' for i in range(30)] + ['RandomForest_Result', 'DenseNet_Result']

if not file_path.exists():
    df = pd.DataFrame(columns=columns)
    df.to_excel(file_path, index=False)

@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    conn = sqlite3.connect(BASE_DIR / 'users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return jsonify({'success': True, 'username': username, 'redirect': '/breast_cancer.html'})
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials!'})

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Both fields are required!'})
    
    conn = sqlite3.connect(BASE_DIR / 'users.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Registration successful!'})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'success': False, 'message': 'Username already exists!'})

@app.route('/breast_cancer.html')
def breast_cancer():
    return send_from_directory(BASE_DIR, 'breast_cancer.html')

@app.route('/test_data.html')
def test_data():
    return send_from_directory(BASE_DIR, 'test_data.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    username = data.get('username', 'Guest')  # Default to 'Guest' if username is not provided
    
    # Extract features
    features = [data.get(f'feature_{i}', 0) for i in range(30)]
    
    if None in features:
        return jsonify({'success': False, 'message': 'Invalid input values!'})

    # Perform predictions
    rf_prediction = rf_model.predict([features])[0]
    densenet_prediction = densenet_model.predict([features])[0]
    densenet_result = 'Malignant' if densenet_prediction > 0.5 else 'Benign'

    result = {
        'RandomForest': 'Malignant' if rf_prediction == 1 else 'Benign',
        'DenseNet': densenet_result
    }
    
    # Append data to Excel file
    df = pd.DataFrame([[
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        username
    ] + features + [result['RandomForest'], result['DenseNet']]], 
        columns=columns)
    
    if file_path.exists():
        existing_df = pd.read_excel(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_excel(file_path, index=False)

    return jsonify({'success': True, 'prediction': result})

@app.route('/test_results.xlsx')
def download_results():
    return send_from_directory(BASE_DIR, 'test_results.xlsx')

# Serve Static Files (CSS, Images, JS)
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
