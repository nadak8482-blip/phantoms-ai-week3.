import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run_model_comparison():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    lr_model = LogisticRegression(max_iter=5000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
    
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
    
    knn_unscaled = KNeighborsClassifier(n_neighbors=5)
    knn_unscaled.fit(X_train, y_train)
    knn_unscaled_acc = accuracy_score(y_test, knn_unscaled.predict(X_test))
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_scaled.fit(X_train_scaled, y_train)
    knn_scaled_acc = accuracy_score(y_test, knn_scaled.predict(X_test))
    
    comparison_table = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree Classifier",
            "KNN (Unscaled)",
            "KNN (Scaled)"
        ],
        "Accuracy": [
            lr_acc,
            dt_acc,
            knn_unscaled_acc,
            knn_scaled_acc
        ]
    })
    
    print(comparison_table.to_string(index=False))

if __name__ == "__main__":
    run_model_comparison()
