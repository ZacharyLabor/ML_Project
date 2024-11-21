from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load models
nb_model = joblib.load('models/naive_bayes_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
retailwise_model = joblib.load('models/retailwise_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/naive_bayes', methods=['GET', 'POST'])
def naive_bayes():
    if request.method == 'POST':
        try:
            # Get user input
            income = float(request.form.get('income'))
            spending_score = float(request.form.get('spending_score'))

            # Validate input
            if income <= 0 or spending_score < 0 or spending_score > 100:
                return render_template('naive_bayes.html', error="Invalid input! Please provide valid values.")

            # Prepare data
            input_data = pd.DataFrame({'Income': [income], 'SpendingScore': [spending_score]})

            # Make prediction
            predicted_class = nb_model.predict(input_data)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]

            # Display prediction
            return render_template('naive_bayes.html', prediction=predicted_label, income=income, spending_score=spending_score)

        except Exception as e:
            return render_template('naive_bayes.html', error=f"Error occurred: {str(e)}")

    return render_template('naive_bayes.html')

@app.route('/retailwise', methods=['GET', 'POST'])
def retailwise():
    if request.method == 'POST':
        try:
            # Get user input
            feature_1 = float(request.form.get('feature_1'))
            feature_2 = float(request.form.get('feature_2'))

            # Prepare data
            input_data = pd.DataFrame({'Feature1': [feature_1], 'Feature2': [feature_2]})

            # Make prediction
            predicted_output = retailwise_model.predict(input_data)[0]

            # Display prediction
            return render_template('retailwise.html', prediction=predicted_output, feature_1=feature_1, feature_2=feature_2)

        except Exception as e:
            return render_template('retailwise.html', error=f"Error occurred: {str(e)}")

    return render_template('retailwise.html')

@app.route('/label_encoder')
def label_encoder_page():
    return render_template('label_encoder.html')

if __name__ == '__main__':
    app.run(debug=True)

