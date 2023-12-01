from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib

app = Flask(__name__)

# Load the pickled model
with open('loan_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the pickled scaler
scaler = joblib.load('scale_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        no_of_dependents = float(request.form['no_of_dependents'])
        education = int(request.form['education'])
        self_employed = float(request.form['self_employed'])
        income_annum = int(request.form['income_annum'])
        loan_amount = int(request.form['loan_amount'])
        loan_term = int(request.form['loan_term'])
        cibil_score = int(request.form['cibil_score'])
        residential_assets_value = int(request.form['residential_assets_value'])
        commercial_assets_value = int(request.form['commercial_assets_value'])
        luxury_assets_value = int(request.form['luxury_assets_value'])
        bank_asset_value = int(request.form['bank_asset_value'])

        # Scale the input data
        input_data = np.array([[no_of_dependents, education, self_employed, income_annum, loan_amount,loan_term, cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value]])
        scaled_data = scaler.transform(input_data)

        # Make a prediction using the loaded model
        prediction = model.predict(scaled_data)[0]

        result_label = 'Congrats! Your loan got approved' if prediction == 0 else "Sorry, your lan didn't get approved"

        return render_template('result.html', prediction=result_label)

if __name__ == '__main__':
    app.run(debug=True)
