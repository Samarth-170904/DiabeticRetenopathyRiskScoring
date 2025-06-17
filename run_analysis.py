import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("Libraries imported successfully.")

def clean_data(df):
    print("Cleaning dataset...")
    
    

def main():
    raw_df = pd.read_csv('metadata.csv')
    print(f"Dataset loaded successfully, having columns and rows:{df.shape}")
    
    df = clean_data(raw_df)
    
    
    
if __name__ == '__main__':
    main()