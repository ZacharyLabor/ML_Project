import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import joblib  # For saving models and encoders

# Step 1: Load the dataset
file_path = 'customer_segmentation_dataset.csv'  # Update this if needed
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
    exit()

# Step 2: Preprocess the data
# Check if required columns are present
required_columns = ['CustomerID', 'Income', 'SpendingScore', 'PurchasingHabit']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"Error: Missing required columns: {missing_columns}")
    exit()

# Preserve CustomerID for identification (optional)
customer_ids = data['CustomerID']

# Use 'Income' and 'SpendingScore' as features
X = data[['Income', 'SpendingScore']]  # Features
y = data['PurchasingHabit']  # Target

# Encode the target variable ('PurchasingHabit')
le = LabelEncoder()
y = le.fit_transform(y)  # Encode 'Low', 'Medium', 'High' as numerical values

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
print("Naive Bayes model trained successfully!")

# Step 5: Train the Brand Name Encoder
brand_name_data = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']  # Example data
brand_name_encoder = LabelEncoder()
brand_name_encoder.fit(brand_name_data)
print("Brand Name Encoder trained successfully!")

# Step 6: Train the Product Category Encoder and Model
product_category_data = ['Electronics', 'Clothing', 'Groceries', 'Furniture', 'Books']  # Example categories
product_category_encoder = LabelEncoder()
encoded_categories = product_category_encoder.fit_transform(product_category_data)

# Prepare synthetic data for product category model
product_features = pd.DataFrame({
    'Feature1': [10, 20, 15, 25, 5],
    'Feature2': [30, 40, 35, 50, 10]
})
product_category_model = GaussianNB()
product_category_model.fit(product_features, encoded_categories)
print("Product Category Model and Encoder trained successfully!")

# Step 7: Make predictions using the Naive Bayes model
y_pred = nb_model.predict(X_test)
predicted_labels = le.inverse_transform(y_pred)

# Step 8: Combine the predictions with features and CustomerID into a DataFrame
output_df = X_test.copy()
output_df['PredictedPurchasingHabit'] = predicted_labels
output_df['CustomerID'] = customer_ids.iloc[X_test.index].values

# Add a sequential "Customer" column for display
output_df.reset_index(drop=True, inplace=True)
output_df['Customer'] = range(1, len(output_df) + 1)

# Step 9: Save all models and encoders for future use
joblib.dump(nb_model, 'models/naive_bayes_model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(brand_name_encoder, 'models/brand_name_encoder.pkl')
joblib.dump(product_category_encoder, 'models/product_category_encoder.pkl')
joblib.dump(product_category_model, 'models/product_category_model.pkl')
print("All models and encoders saved successfully!")

# Step 10: Display sample results for Naive Bayes predictions
print("\nSample Predictions:")
print(output_df[['Customer', 'CustomerID', 'Income', 'SpendingScore', 'PredictedPurchasingHabit']].head())

# Optional: Save Naive Bayes predictions to a CSV file
output_df.to_csv('predicted_purchasing_habits.csv', index=False)
print("\nNaive Bayes predictions saved to 'predicted_purchasing_habits.csv'")

# Step 11: Test Brand Name Encoder
test_brand = 'BrandB'
encoded_brand = brand_name_encoder.transform([test_brand])[0]
print(f"\nTest Brand Encoding: {test_brand} -> {encoded_brand}")

# Step 12: Test Product Category Model
test_features = pd.DataFrame({'Feature1': [18], 'Feature2': [38]})
predicted_category = product_category_model.predict(test_features)[0]
decoded_category = product_category_encoder.inverse_transform([predicted_category])[0]
print(f"\nTest Product Category Prediction: {test_features.values} -> {decoded_category}")
