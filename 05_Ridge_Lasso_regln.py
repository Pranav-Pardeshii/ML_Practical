
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = load_diabetes()
X, y = data.data, data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 5. Ridge Regression
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# 6. Lasso with different alpha values
alphas = [0.01, 0.1, 1.0]

for a in alphas:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_train, y_train)

    y_pred = lasso.predict(X_test)

    print("\nLasso Alpha:", a)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
    print("Zero Coefficients:", np.sum(lasso.coef_ == 0))

# 7. Evaluate Linear and Ridge
print("\nLinear Regression")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

print("\nRidge Regression")
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R2 Score:", r2_score(y_test, y_pred_ridge))

# 8. Coefficient comparison (Lasso alpha = 0.1)
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train, y_train)

x = np.arange(10)

plt.bar(x-0.25, lr.coef_, 0.25, label="Linear")
plt.bar(x, ridge.coef_, 0.25, label="Ridge")
plt.bar(x+0.25, lasso.coef_, 0.25, label="Lasso")

plt.xticks(x, data.feature_names)
plt.xlabel("Features")
plt.ylabel("Coefficient Values")
plt.title("Linear, Ridge and Lasso Comparison")
plt.legend()
plt.grid()
plt.show()
