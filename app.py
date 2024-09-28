from flask import Flask, request, jsonify
import numpy as np
from joblib import load
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable or default to 5000
    app.run(host='0.0.0.0', port=port)

# Load pre-trained models
lr_model = load('lr_model.pkl')
svm_model = load('svm_model.pkl')
knn_model = load('knn_model.pkl')
rf_model = load('rf_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        data['Glucose'], data['BloodPressure'], data['SkinThickness'],
        data['Insulin'], data['BMI'], data['Age'], data['Pregnancies']
    ]])

    # Select the model based on user input
    model_choice = data.get('model', 'lr')  # Default is Logistic Regression

    if model_choice == 'svm':
        prediction = svm_model.predict(features)
    elif model_choice == 'knn':
        prediction = knn_model.predict(features)
    elif model_choice == 'rf':
        prediction = rf_model.predict(features)
    else:
        prediction = lr_model.predict(features)

    result = 'Diabetic Retinopathy Detected' if prediction[0] == 1 else 'No Diabetic Retinopathy'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
