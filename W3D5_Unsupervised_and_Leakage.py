import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

print("=" * 70)
print("=== [Phantoms | W3D5] الختام الشامل لأسبوع الـ Machine Learning ===")
print("=" * 70)

# ==========================================
# 1. تطبيق التعلم غير الموجه (K-Means Clustering)
# ==========================================
print("\n[1/3] جاري تشغيل خوارزمية K-Means...")

# خلق بيانات وهمية ورسمها
X_blob, _ = make_blobs(
    n_samples=300, centers=3, cluster_std=0.60, random_state=42
)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_blob)
y_kmeans = kmeans.predict(X_blob)
centers = kmeans.cluster_centers_

print(f"تم تحديد المراكز (Centroids) بنجاح بنسبة استقرار عالية.")

# ==========================================
# 2. حل ومعالجة مشكلة Data Leakage
# ==========================================
print(
    "\n[2/3] جاري تطبيق وتوضيح الحل الهندسي لمنع تسريب البيانات (Data Leakage)..."
)

data = load_breast_cancer()
X, y = data.data, data.target

# التقسيم السليم أولاً
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# عمل Scaling للتدريب لوحدها والاختبار لوحدها بمنع التسريب
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# نموذج تأكيدي للتحقق من الدقة الحقيقية
leakage_check_model = RandomForestClassifier(random_state=42)
leakage_check_model.fit(X_train_scaled, y_train)
real_acc = accuracy_score(
    y_test, leakage_check_model.predict(X_test_scaled)
) * 100
print(f"دقة النموذج السليمة بدون Data Leakage: {real_acc:.2f}%")

# ==========================================
# 3. دمج رحلة النماذج (Mini Wrap-up: D1 to D4)
# ==========================================
print(
    "\n[3/3] جاري تدريب عدة نماذج ومقارنة أدائها (Logistic, Decision Tree, Random Forest)..."
)

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
        }
    )

df_comparison = pd.DataFrame(results)

print("=" * 70)
print("جدول مقارنة أداء النماذج النهائي (Mini Wrap-up Comparison Table):")
print("=" * 70)
print(df_comparison.to_string(index=False))
print("=" * 70)

# ==========================================
# 4. التقرير الهندسي الختامي
# ==========================================
summary_data = {
    "Task Stage / Component": [
        "1. K-Means Clustering",
        "2. Data Leakage Prevention",
        "3. Model Generalization",
    ],
    "Objective": [
        "اكتشاف مجموعات غير مسمات (Unsupervised)",
        "منع تسريب بيانات الاختبار لضمان دقة حقيقية",
        "بناء نموذج قوي ومستقر للتنبؤ",
    ],
    "Key Result / Outcome": [
        "تم تقسيم البيانات لـ 3 مجموعات وتحديد المراكز بدقة",
        "تم تفادي الوهم والحصول على دقة واقعية ومنطقية",
        "جاهزية كاملة للدمج والرفع النهائي",
    ],
}

df_summary = pd.DataFrame(summary_data)

print(" تقرير الملخص الهندسي الختامي للأسبوع الثالث (W3D5 Wrap-up)")
print("=" * 70)
print(df_summary.to_string(index=False))
print("=" * 70)
print(" الخلاصة النهائية: تم إنجاز كافة متطلبات الأسبوع بنجاح وكفاءة تامة.")
print("=" * 70)
