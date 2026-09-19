import numpy as np
import pandas as pd
import seaborn as sns

def initialize_numpy_operations():
    data_array = np.array([12, 24, 36, 48, 60, 72, 84])
    
    mean_val = np.mean(data_array)
    std_val = np.std(data_array)
    
    print("Generated Array:", data_array)
    print("Calculated Mean:", round(mean_val, 2))
    print("Calculated Standard Deviation:", round(std_val, 2))
    print()

def initialize_pandas_dataset():
    df = sns.load_dataset('iris')
    
    print("First 5 Rows of Iris Dataset:")
    print(df.head())
    print()
    
    print("Dataset Columns:")
    print(list(df.columns))
    print()
    
    return df

def ai_metadata_analysis(df):
    analysis_report = """
    Analysis Report:
    - Selected Dataset: Iris Dataset
    - Target Column: 'species' (contains explicit class labels: setosa, versicolor, virginica)
    - Classification Type: Supervised Learning
    - Justification: The dataset qualifies as a Supervised Learning problem because 
      it includes historical target labels ('species') mapped directly against feature 
      attributes (sepal and petal dimensions). The model utilizes these known labels 
      during training to classify new instances accurately.
    """
    print(analysis_report)

if __name__ == "__main__":
    print("System Initiated: W3D1 Machine Learning Pipeline Running...")
    print()
    initialize_numpy_operations()
    dataset = initialize_pandas_dataset()
    ai_metadata_analysis(dataset)
    print("System Status: Task execution completed successfully.")
