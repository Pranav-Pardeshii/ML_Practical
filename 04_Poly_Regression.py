import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load data
df = pd.read_csv("mpg.csv").dropna(subset=['horsepower'])
X = df[['horsepower']]
y = df['mpg']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Test poly degrees from 1 to 10
degrees = range(1, 11)
train_rmse, test_rmse = [],[]

for d in degrees:
    # Step A: Create poly columns
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Setp B: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    # Step C: Fit linear regression on polynomial features
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Setp D: Record test and test RMSE
    train_rmse.append(np.sqrt(mean_squared_error(y_train, model.predict(X_train_scaled))))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled))))
    print(f"Degree {d:2d} -> Train RMSE: {train_rmse[-1]:.2f} | Test RMSE: {test_rmse[-1]:.2f}")


# 4. Plot 1: Train vs Test RMSE
plt.plot(degrees, train_rmse, label="Train RMSE")
plt.plot(degrees, test_rmse, label="Test RMSE")

plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("Bias Variance Tradeoff")
plt.legend()
plt.grid()
plt.show()

# 5. Plot 2: Polynomial Curves
plt.scatter(X_train, y_train)

x_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

for d in [1, 2, 10]:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    scaler = StandardScaler()

    X_poly = poly.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_poly)

    model = LinearRegression()
    model.fit(X_scaled, y_train)

    y_grid = model.predict(
        scaler.transform(poly.transform(x_grid))
    )

    plt.plot(x_grid, y_grid, label=f"Degree {d}")

plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression")
plt.legend()
plt.grid()
plt.show()
