import os
import logging
import warnings
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import numpy as np
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress XGBoost warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define feature names
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load model and scaler
try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    logger.info(f"Loading scaler from: {SCALER_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return "Error loading page", 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        logger.error(f"Error serving static file: {str(e)}")
        return "File not found", 404

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.form.to_dict()
        logger.info(f"Received prediction request with data: {data}")
        
        # Validate input
        for field in FEATURE_NAMES:
            if field not in data:
                error_msg = f'Missing field: {field}'
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 400
            try:
                data[field] = float(data[field])
            except ValueError:
                error_msg = f'Invalid value for {field}. Please enter a valid number.'
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 400
        
        # Prepare input data
        input_data = np.array([data[field] for field in FEATURE_NAMES]).reshape(1, -1)
        logger.info(f"Prepared input data: {input_data}")
        
        # Scale input data
        try:
            scaled_data = scaler.transform(input_data)
            logger.info(f"Scaled data: {scaled_data}")
        except Exception as e:
            error_msg = f"Error scaling input data: {str(e)}"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500
        
        # Make prediction
        try:
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0]
            logger.info(f"Prediction: {prediction}, Probability: {probability}")
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
            else:
                # For models without feature_importances_, use coefficients
                if hasattr(model, 'coef_'):
                    feature_importance = dict(zip(FEATURE_NAMES, abs(model.coef_[0])))
                else:
                    feature_importance = {name: 1.0/len(FEATURE_NAMES) for name in FEATURE_NAMES}
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            error_msg = f"Error making prediction: {str(e)}"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500
        
        # Format response
        result = {
            'prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability': {
                'diabetic': float(probability[1]),
                'non_diabetic': float(probability[0])
            },
            'feature_importance': [
                {
                    'feature': feature,
                    'importance': float(importance)
                } for feature, importance in sorted_features
            ]
        }
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Unexpected error during prediction: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
