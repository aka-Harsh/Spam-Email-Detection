# app.py - Modified version

from flask import Flask, render_template, request, jsonify
import os
import pickle
from spam_detector import EmailSpamDetector

app = Flask(__name__)

# Initialize the spam detector
detector = EmailSpamDetector()

model_path = 'spam_detector_model.pkl'  
vectorizer_path = 'tfidf_vectorizer.pkl'

print(f"Looking for model at: {os.path.abspath(model_path)}")
model_loaded = False

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    try:
        detector.load_model(model_path, vectorizer_path)
        model_loaded = True
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model files not found. Looking for: {model_path} and {vectorizer_path}")
    print("Please run main.py to train the model first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_content = request.form['email_content']
        
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            })
        
        result = detector.predict_email(email_content)
        
        response = {
            'prediction': result['prediction'],
            'confidence': round(float(result['confidence']), 4) if result['confidence'] is not None else None,
            'status': 'success'
        }
        
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)