import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the SVM model trained for diabetes prediction
model = pickle.load(open('model/prac_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = [float(request.form[field]) for field in ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
    user_input = np.array(user_input).reshape(1, -1)
    prediction = model.predict(user_input)
    result = 'Diabetic' if prediction == 1 else 'Not Diabetic'

    return render_template('index.html', prediction_output=f'The Person is {result}')

if __name__ == "__main__":
    app.run(debug=True)

