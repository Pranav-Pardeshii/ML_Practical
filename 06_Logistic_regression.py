import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score, 
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# 1. Load Breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

#5. Make prediction
y_pred = model.predict(X_test_scaled)

# Predict probabilities for ROC curve
y_prob = model.predict_proba(X_test_scaled)[:,1]

# 6. Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("Classification Report:\n",
      classification_report(y_test, y_pred))

# 7. Plot ROC curve
fpr, tpr, threshold = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], 'r--', label="random guess")

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve")

plt.legend()
plt.grid()
plt.show()