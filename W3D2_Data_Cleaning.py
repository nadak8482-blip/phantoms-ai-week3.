import pandas as pd
import numpy as np

def load_and_explore_data():
    print("Step 1: Loading and Exploring Dataset")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    print("Dataset Info:")
    print(df.info())
    print("\nStatistical Description:")
    print(df.describe())
    print("\nMissing Values Count per Column:")
    print(df.isnull().sum())
    return df

def clean_data(df):
    print("\nStep 2: Handling Missing Values & Data Cleaning")
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df = df.dropna(subset=['Embarked'])
    df = df.drop(columns=['Cabin'])
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    print("Missing values after cleaning:")
    print(df.isnull().sum())
    return df

def feature_engineering_and_encoding(df):
    print("\nStep 3: Feature Engineering and Categorical Conversion")
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df

def define_features_and_labels(df):
    print("\nStep 4: Defining Features and Labels")
    y = df['Survived']
    features_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Embarked_Q', 'Embarked_S']
    X = df[features_columns]
    print("Features preview (X):")
    print(X.head(3))
    print("\nLabel preview (y):")
    print(y.head(3))
    return df

if __name__ == "__main__":
    print("Pipeline Started: W3D2 Data Cleaning & Preparation...")
    df = load_and_explore_data()
    df_cleaned = clean_data(df)
    df_processed = feature_engineering_and_encoding(df_cleaned)
    define_features_and_labels(df_processed)
    output_filename = "cleaned_data.csv"
    df_processed.to_csv(output_filename, index=False)
    print(f"\nPipeline Finished Successfully! Cleaned data saved to {output_filename}")
