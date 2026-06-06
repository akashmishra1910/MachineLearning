import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Use Classification metrics instead of Regression metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import data
data = pd.read_csv('app_subscription_data_500.csv')
print("--- First 3 rows of data ---")
print(data.head(3))
print("\n")

# Step 1: Define features (X) and target (y)
X = data[['App_Usage_Time_min', 'Number_of_Sessions', 'Features_Used']]
y = data['Purchased_Premium']

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3 & 4: Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on test data
predictions = model.predict(X_test)

# Step 6: Predict subscription for a new user
# (Columns must match the training data X exactly)
new_user_data = pd.DataFrame([{
    'App_Usage_Time_min': 45,
    'Number_of_Sessions': 4,
    'Features_Used': 3
}])
predicted_sub = model.predict(new_user_data)
print("Predicted Subscription for New User (1=Yes, 0=No):", predicted_sub[0])
print("\n")

# Step 7: Model parameters
print("Coefficients (Usage, Sessions, Features):", model.coef_)
print("Intercept:", model.intercept_)
print("\n")

# Step 8: Evaluate the model
print("--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
print("\n")

# Step 9: Show Actual vs Predicted
results = pd.DataFrame({
    'Actual_Subscription': y_test,
    'Predicted_Subscription': predictions
})
print("--- Actual vs Predicted Results ---")
print(results.head(10))
