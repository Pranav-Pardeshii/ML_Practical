import matplotlib.pyplot as plt 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load wine dataset
data = load_wine()
X, y = data.data, data.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Compare different kernels
kernels = ['linear','poly','rbf']
accuracies = {}

for k in kernels:
    svm = SVC(kernel= k, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[k] = accuracy

    print("kernel:",k)
    print("Accuracy:", round(accuracy, 4))
    print("confusion_matrix: ", confusion_matrix(y_test, y_pred))
    print()

# 5. Plot accuracy comparision
plt.bar(accuracies.keys(), accuracies.values())

plt.xlabel("SVM kernels")
plt.ylabel("Accuracy")
plt.title("SVM kernel comparision")

plt.ylim(0.8, 1.02)
plt.grid()
plt.show()