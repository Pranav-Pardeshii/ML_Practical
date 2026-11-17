# For 2, 3, 4, 5, 6, 7, 10

import matplotlib.pyplot as plt
from sklearn.datasets import load_xxx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.xxx import ModelName
from sklearn.metrics import metric_name


# 1. Load
data = load_xxx()
X, y = data.data, data.target

# 2. Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Only use stratify for classification -> balances the train and test dataset
)

# 3. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Fit and predict 
model = ModelName()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# 5. Evaluate and plot
print(metric_name(y_test, y_predict))
