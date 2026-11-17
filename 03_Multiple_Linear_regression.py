import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load data
data = load_diabetes()
X, y = data.data, data.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Fit and predict 
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict unseed test patients
y_pred = model.predict(X_test)

# 5. Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.2f}")
print("\nFeature Coefficients (weights):")
for name, coef in zip(data.feature_names, model.coef_):
    print(f"  {name:5s}: {coef:.2f}")

# 6. Plot actual vs Predicted values
# 6. Plot Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.grid()
plt.show()