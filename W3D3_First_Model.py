
from google.colab import files
import pandas as pd
import glob

csv_files = glob.glob('*.csv')

if csv_files:
    file_name = csv_files[0]
    print(f"تم استخدام الملف بنجاح: {file_name}")
    df = pd.read_csv(file_name)
    
    target_column = 'Survived' 

    X = df.drop(columns=[target_column]) 
    y = df[target_column] 

    X = X.select_dtypes(include=['number'])
    
    X = X.fillna(X.mean())

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"حجم بيانات التدريب: {X_train.shape[0]} صف")
    print(f"حجم بيانات الاختبار: {X_test.shape[0]} صف\n")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("تم تدريب النموذج بنجاح! 🎉")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nدقة النموذج (Accuracy): {accuracy * 100:.2f}%\n")
else:
    print("من فضلك تأكد من رفع ملف CSV في بيئة العمل.")