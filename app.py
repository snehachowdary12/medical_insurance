from flask import Flask, render_template, request
import pickle
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Load trained models
with open("models/linear_regression.pkl", "rb") as f:
    linear_model = pickle.load(f)
with open("models/random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("models/xgboost.pkl", "rb") as f:
    xgb_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['Age']),
            float(request.form['Diabetes']),
            float(request.form['BloodPressureProblems']),
            float(request.form['AnyTransplants']),
            float(request.form['AnyChronicDiseases']),
            float(request.form['Height']),
            float(request.form['Weight']),
            float(request.form['KnownAllergies']),
            float(request.form['HistoryOfCancerInFamily']),
            float(request.form['NumberOfMajorSurgeries'])
        ]

        features = np.array([features])

        pred_linear = linear_model.predict(features)[0]
        pred_rf = rf_model.predict(features)[0]
        pred_xgb = xgb_model.predict(features)[0]

        return render_template('index.html', 
                               prediction_linear=round(pred_linear, 2), 
                               prediction_rf=round(pred_rf, 2), 
                               prediction_xgb=round(pred_xgb, 2))
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
