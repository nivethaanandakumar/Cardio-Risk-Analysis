import pandas as pd
import joblib

# Load the trained model
model = joblib.load('cardio_model.pkl')

# Load the new data (make sure the new data has the same structure as the training data)
new_data = pd.read_csv('heart.csv')  

# Step 1: Replace '-' with NaN
new_data.replace('-', pd.NA, inplace=True)

# Step 2: Convert all columns to numeric where possible
for col in new_data.columns:
    new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

# Step 3: Handle missing values (fill with mean of each column)
new_data.fillna(new_data.mean(), inplace=True)

# Step 4: Drop the 'target' column if it exists in the new data
new_data = new_data.drop('target', axis=1, errors='ignore')  # Ignore if 'target' is not there

# Step 5: Ensure the new data has the same columns as the training data
# (This assumes you saved the feature names during training and have the same features in the new data)
X_train_columns = joblib.load('cardio_model.pkl').feature_names_in_
new_data = new_data[X_train_columns]

# Step 6: Make predictions
predictions = model.predict(new_data)

# Output the predictions
print(predictions)
