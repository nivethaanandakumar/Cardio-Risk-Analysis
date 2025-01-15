import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Load the dataset
df = pd.read_csv('heart.csv')

# Data preprocessing
df.replace('-', pd.NA, inplace=True)  # Replace '-' with NaN
df = df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric where possible
df.fillna(df.mean(), inplace=True)  # Fill NaN with column means

# Separate features and target
X = df.drop('target', axis=1)  # Drop target column
y = df['target']

# Handle class imbalance
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights_dict)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
joblib.dump(model, 'cardio_model.pkl')
