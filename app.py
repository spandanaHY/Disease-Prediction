from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'disease_prediction_secret_2024'

# Simple user storage (in memory)
users = {}

# Load models and scalers
diabetes_model  = pickle.load(open('diabetes_model.pkl', 'rb'))
diabetes_scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))
heart_model     = pickle.load(open('heart_model.pkl', 'rb'))
heart_scaler    = pickle.load(open('heart_scaler.pkl', 'rb'))

# ── AUTH ──────────────────────────────────────────────────

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        email    = request.form['email'].strip()
        password = request.form['password']
        if username in users:
            error = 'Username already exists.'
        elif not username or not email or not password:
            error = 'All fields are required.'
        else:
            users[username] = {'email': email, 'password': password}
            session['user'] = username
            return redirect(url_for('about'))
    return render_template('register.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['user'] = username
            return redirect(url_for('about'))
        else:
            error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── PAGES ─────────────────────────────────────────────────

@app.route('/about')
def about():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('about.html', user=session['user'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    result      = None
    disease     = None
    risk        = None
    input_data  = None
    error       = None

    if request.method == 'POST':
        disease = request.form.get('disease')
        try:
            if disease == 'diabetes':
                features = [
                    float(request.form['Pregnancies']),
                    float(request.form['Glucose']),
                    float(request.form['BloodPressure']),
                    float(request.form['SkinThickness']),
                    float(request.form['Insulin']),
                    float(request.form['BMI']),
                    float(request.form['DiabetesPedigreeFunction']),
                    float(request.form['Age']),
                ]
                scaled = diabetes_scaler.transform([features])
                pred   = diabetes_model.predict(scaled)[0]
                proba  = diabetes_model.predict_proba(scaled)[0] if hasattr(diabetes_model, 'predict_proba') else None
                result = 'Diabetes Detected' if pred == 1 else 'No Diabetes'
                risk   = 'High Risk' if pred == 1 else 'Low Risk'
                conf   = round(max(proba)*100, 1) if proba is not None else None

            elif disease == 'heart':
                features = [
                    float(request.form['age']),
                    float(request.form['sex']),
                    float(request.form['cp']),
                    float(request.form['trestbps']),
                    float(request.form['chol']),
                    float(request.form['fbs']),
                    float(request.form['restecg']),
                    float(request.form['thalach']),
                    float(request.form['exang']),
                    float(request.form['oldpeak']),
                    float(request.form['slope']),
                    float(request.form['ca']),
                    float(request.form['thal']),
                ]
                scaled = heart_scaler.transform([features])
                pred   = heart_model.predict(scaled)[0]
                proba  = heart_model.predict_proba(scaled)[0] if hasattr(heart_model, 'predict_proba') else None
                result = 'Heart Disease Detected' if pred == 1 else 'No Heart Disease'
                risk   = 'High Risk' if pred == 1 else 'Low Risk'
                conf   = round(max(proba)*100, 1) if proba is not None else None

            input_data = request.form
        except Exception as e:
            error = f'Please fill all fields correctly. ({str(e)})'

    return render_template('predict.html',
                           user=session['user'],
                           result=result,
                           disease=disease,
                           risk=risk,
                           error=error,
                           input_data=input_data)

@app.route('/graphs')
def graphs():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('graphs.html', user=session['user'])

if __name__ == '__main__':
    app.run(debug=True)