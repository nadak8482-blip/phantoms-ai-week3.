import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
clusters = kmeans.labels_

print(kmeans.cluster_centers_)
print(clusters[:10])

np.random.seed(42)
X_bad = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)
X_bad[:, 0] = y + np.random.normal(0, 0.01, 100) 

scaler_wrong = StandardScaler()
X_wrong_scaled = scaler_wrong.fit_transform(X_bad)

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_wrong_scaled, y, test_size=0.2, random_state=42)
model_wrong = LogisticRegression()
model_wrong.fit(X_train_w, y_train_w)
print(accuracy_score(y_test_w, model_wrong.predict(X_test_w)))

X_train_right, X_test_right, y_train_right, y_test_right = train_test_split(X_bad, y, test_size=0.2, random_state=42)

scaler_right = StandardScaler()
X_train_scaled = scaler_right.fit_transform(X_train_right)
X_test_scaled = scaler_right.transform(X_test_right)

model_right = LogisticRegression()
model_right.fit(X_train_scaled, y_train_right)
print(accuracy_score(y_test_right, model_right.predict(X_test_scaled)))
