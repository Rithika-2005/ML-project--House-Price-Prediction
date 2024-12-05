import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('house_prices.csv')  # Replace with your dataset file
X = data[['size', 'bedrooms']]  # Features
y = data['price']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Plot results
plt.scatter(X_test['size'], y_test, color='blue', label='Actual')
plt.scatter(X_test['size'], y_pred, color='red', label='Predicted')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
