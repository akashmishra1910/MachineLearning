import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#import data
data=pd.read_csv('delivery_time_dataset_500.csv')
print(data.head(3))
# Step 1: Define features (X) and target (y)
#For a multivariate model, include multiple feature columns
X = data[['Distance_KM', 'Traffic_Signals', 'Order_Weight_KG']]
y = data['Delivery_Time_Minutes']
# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Step 3: Create the model
model = LinearRegression()
# Step 4: Train the model
model.fit(X_train, y_train)
# Step 5: Make predictions on test data
predictions = model.predict(X_test)
print("Actual Time:\n", y_test)
print("Predicted Time:\n", predictions)
# Step 6: Predict price for a new order
# For multivariate prediction, provide values for all features
new_order_time = pd.DataFrame([{
    'Distance_KM': 12,
    'Traffic_Signals': 3,
    'Order_Weight_KG': 13
}])
predicted_time = model.predict(new_order_time)
print("Predicted Time for New Order:", predicted_time)
# Step 7: Model parameters
print("\nCoefficients (m):", model.coef_)
print("Intercept (b):", model.intercept_)
# Step 8: Evaluate the model

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nMean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)
sns.pairplot(data)
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions
})

print(results.head(10))



