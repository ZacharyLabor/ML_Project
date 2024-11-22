from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load models
nb_model = joblib.load('models/naive_bayes_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
retailwise_model = joblib.load('models/retailwise_model.pkl')
brand_name_encoder = joblib.load('models/brand_name_encoder.pkl')
product_category_encoder = joblib.load('models/product_category_encoder.pkl')
product_category_model = joblib.load('models/product_category_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

# Naive Bayes Predictor Route
@app.route('/naive_bayes', methods=['GET', 'POST'])
def naive_bayes():
    if request.method == 'POST':
        try:
            income = float(request.form.get('income'))
            spending_score = float(request.form.get('spending_score'))

            if income <= 0 or spending_score < 0 or spending_score > 100:
                return render_template('naive_bayes.html', error="Invalid input! Please provide valid values.")

            input_data = pd.DataFrame({'Income': [income], 'SpendingScore': [spending_score]})
            predicted_class = nb_model.predict(input_data)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]

            return render_template('naive_bayes.html', prediction=predicted_label, income=income, spending_score=spending_score)
        except Exception as e:
            return render_template('naive_bayes.html', error=f"Error occurred: {str(e)}")

    return render_template('naive_bayes.html')

# RetailWise Model Route
@app.route('/retailwise', methods=['GET', 'POST'])
def retailwise():
    if request.method == 'POST':
        try:
            feature_1 = float(request.form.get('feature_1'))
            feature_2 = float(request.form.get('feature_2'))

            input_data = pd.DataFrame({'Feature1': [feature_1], 'Feature2': [feature_2]})
            predicted_output = retailwise_model.predict(input_data)[0]

            return render_template('retailwise.html', prediction=predicted_output, feature_1=feature_1, feature_2=feature_2)
        except Exception as e:
            return render_template('retailwise.html', error=f"Error occurred: {str(e)}")

    return render_template('retailwise.html')

# Label Encoder Route
@app.route('/label_encoder')
def label_encoder_page():
    return render_template('label_encoder.html')

# Brand Name Encoder Route
@app.route('/brand_name_encoder', methods=['GET', 'POST'])
def brand_name_encoder_page():
    if request.method == 'POST':
        try:
            brand_name = request.form.get('brand_name')
            if not brand_name:
                return render_template('brand_name_encoder.html', error="Please enter a valid brand name.")

            encoded_brand = brand_name_encoder.transform([brand_name])[0]
            return render_template('brand_name_encoder.html', encoded_brand=encoded_brand, brand_name=brand_name)
        except Exception as e:
            return render_template('brand_name_encoder.html', error=f"Error occurred: {str(e)}")

    return render_template('brand_name_encoder.html')

# Product Category Encoder Route
@app.route('/product_category_encoder', methods=['GET', 'POST'])
def product_category_encoder_page():
    if request.method == 'POST':
        try:
            product_category = request.form.get('product_category')
            if not product_category:
                return render_template('product_category_encoder.html', error="Please enter a valid product category.")

            encoded_category = product_category_encoder.transform([product_category])[0]
            return render_template('product_category_encoder.html', encoded_category=encoded_category, product_category=product_category)
        except Exception as e:
            return render_template('product_category_encoder.html', error=f"Error occurred: {str(e)}")

    return render_template('product_category_encoder.html')

# Product Category Model Route
@app.route('/product_category_model', methods=['GET', 'POST'])
def product_category_model_page():
    if request.method == 'POST':
        try:
            feature1 = float(request.form.get('feature1'))
            feature2 = float(request.form.get('feature2'))

            input_data = pd.DataFrame({'Feature1': [feature1], 'Feature2': [feature2]})
            predicted_category = product_category_model.predict(input_data)[0]

            return render_template('product_category_model.html', predicted_category=predicted_category, feature1=feature1, feature2=feature2)
        except Exception as e:
            return render_template('product_category_model.html', error=f"Error occurred: {str(e)}")

    return render_template('product_category_model.html')

if __name__ == '__main__':
    app.run(debug=True)


