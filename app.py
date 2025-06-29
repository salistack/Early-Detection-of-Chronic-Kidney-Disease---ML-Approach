from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all 24 input features in the same order as training
        features = [
            float(request.form['age']),
            float(request.form['blood_pressure']),
            float(request.form['specific_gravity']),
            float(request.form['albumin']),
            float(request.form['sugar']),
            int(request.form['red_blood_cells']),
            int(request.form['pus_cell']),
            int(request.form['pus_cell_clumps']),
            int(request.form['bacteria']),
            float(request.form['blood_glucose_random']),
            float(request.form['blood_urea']),
            float(request.form['serum_creatinine']),
            float(request.form['sodium']),
            float(request.form['potassium']),
            float(request.form['hemoglobin']),
            float(request.form['packed_cell_volume']),
            float(request.form['white_blood_cell_count']),
            float(request.form['red_blood_cell_count']),
            int(request.form['hypertension']),
            int(request.form['diabetes_mellitus']),
            int(request.form['coronary_artery_disease']),
            int(request.form['appetite']),
            int(request.form['peda_edema']),
            int(request.form['anemia']),
        ]

        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        result = "Yes (CKD)" if prediction == 1 else "No (Healthy)"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
