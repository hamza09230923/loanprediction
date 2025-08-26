from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and label encoders
model = joblib.load('model/loan_prediction_model.joblib')
le_dict = joblib.load('model/label_encoders.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    form_data = request.form.to_dict()

    # Create a DataFrame from the form data
    df = pd.DataFrame([form_data])

    # Preprocess the input
    for col, le in le_dict.items():
        if col in df.columns and col != 'Loan_Status':
            # Use a try-except block to handle unseen labels
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                # If a value is not in the encoder, you might want to handle it
                # For simplicity, we'll assign a default value (e.g., the first class)
                df[col] = 0


    # Convert columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure the order of columns matches the training data
    # (excluding the target variable 'Loan_Status')
    X_train_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                       'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    df = df[X_train_columns]


    # Make prediction
    prediction = model.predict(df)

    # Get the original label for the prediction
    loan_status_le = le_dict['Loan_Status']
    result = loan_status_le.inverse_transform(prediction)[0]

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
