from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('cardio_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from form
        feature_fields = [
            'age', 'sex', 'smoking', 'cholesterol', 'bloodPressure',
            'exercise', 'diabetes', 'familyHistory', 'chestPain',
            'ageAtFirstHeartAttack', 'fatigueLevel', 'maxHeartRateAchieved',
            'sleepDuration'
        ]
        
        features = [float(request.form.get(field, 0)) for field in feature_fields]
        
        # Prepare features for prediction
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        risk = "High Risk" if prediction[0] == 1 else "Low Risk"

    except ValueError as ve:
        risk = f"Input Error: {str(ve)}"
    except Exception as e:
        risk = f"Error: {str(e)}"

    return render_template('result.html', risk=risk)

if __name__ == '__main__':
    app.run(debug=True)
