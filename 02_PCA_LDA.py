# import required libraries
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
data = load_digits()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. Using logistic regression with original dataset
start = time.time()
clf_orig = LogisticRegression(max_iter=1000)
clf_orig.fit(X_train, y_train)
time_orig = time.time() - start
acc_orig = accuracy_score(y_test, clf_orig.predict(X_test))

# Using Logistic regression with PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

start = time.time()
clf_pca = LogisticRegression(max_iter=1000)
clf_pca.fit(X_train_pca, y_train)
time_pca = time.time() - start
acc_pca = accuracy_score(y_test, clf_pca.predict(X_test_pca))

# Using Logistic Regression with LDA
lda = LDA(n_components=9)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

start = time.time()
clf_lda = LogisticRegression(max_iter=1000)
clf_lda.fit(X_train_lda, y_train)
time_lda = time.time() - start
acc_lda = accuracy_score(y_test, clf_lda.predict(X_test_lda))

# Print comparision results
print(f"Original 64 features -> Accuracy: {acc_orig:.4f} | Time: {time_orig:.4f}s")
print(f"PCA ({pca.n_components_}) features) -> Accuracy: {acc_pca:.4f} | Time: {time_pca:.4f}s")
print(f"LDA(9 fetures) -> Accuracy: {acc_lda:.4f} | Time: {time_lda:.4f}s")

# 5. Plot Cumulative Explained Variance for PCA

# Plot PCA Explained Variance
pca_full = PCA().fit(X_train)

plt.plot(pca_full.explained_variance_ratio_.cumsum())
plt.axhline(y=0.95, color='red', linestyle='--')

plt.xlabel("Components")
plt.ylabel("Cumulative Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()
