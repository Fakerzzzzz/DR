import os
from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load pre-trained models (ensure the paths are correct)
lr_model = load('lr_model.pkl')
svm_model = load('svm_model.pkl')
knn_model = load('knn_model.pkl')
rf_model = load('rf_model.pkl')

@app.route('/')
def home():
    return "Diabetic Retinopathy Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the incoming request data
        print("Request Data:", request.json)

        # Extract the data from the request
        data = request.json

        # Check if required fields are present
        required_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'Pregnancies', 'model']
        for field in required_fields:
            if field not in data:
                print(f"Missing field: {field}")
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Create features array and log the features
        features = np.array([[
            data['Glucose'], data['BloodPressure'], data['SkinThickness'],
            data['Insulin'], data['BMI'], data['Age'], data['Pregnancies']
        ]])
        print("Features Array:", features)

        # Select the model and log the selected model
        model_choice = data.get('model', 'lr')
        print(f"Model selected: {model_choice}")

        if model_choice == 'svm':
            prediction = svm_model.predict(features)
        elif model_choice == 'knn':
            prediction = knn_model.predict(features)
        elif model_choice == 'rf':
            prediction = rf_model.predict(features)
        else:
            prediction = lr_model.predict(features)

        # Log the prediction result
        print(f"Prediction result: {prediction}")

        result = 'Diabetic Retinopathy Detected' if prediction[0] == 1 else 'No Diabetic Retinopathy'
        return jsonify({'prediction': result})

    except KeyError as e:
        print(f"KeyError: {e}")
        return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
        # Print the full error message to logs
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
