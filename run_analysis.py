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
    
    df.columns = [col.lower() for col in df.columns]
    df['patient_id'] = df['patient_id'].str.replace('DUP_','')
    df = df.drop_duplicates(subset='patient_id', keep='first')
    print(f"Duplicates removed and shape of the dataset is {df.shape}.")
    
    gender_map = {'M':'M','male':'M','Male':'M','F':'F','female':'F','Female':'F'}
    df.loc[:,'gender'] = df['gender'].str.strip().map(gender_map)
    print("Gender column cleaned")
    
    history_map = {'yes':'Yes','y':'Yes','Y':'Yes','no':'No','n':'No','N':'No','true':'Yes','false':'No','Yes':'Yes','No':'No'}
    df.loc[:,'family_history_dr'] = df['family_history_dr'].str.strip().str.lower().map(history_map)
    print("Family history column cleaned")
    
    smoking_map = {'never': 'Never', 'non-smoker': 'Never', 'former': 'Former', 'ex-smoker': 'Former','current': 'Current', 'smoker': 'Current'}
    df.loc[:,'smoking_status'] = df['smoking_status'].str.strip().str.lower().map(smoking_map)
    print("Smoking status column cleaned")
    
    activity_map = {'low': 'Low', 'sedentary': 'Low', 'moderate': 'Moderate', 'active': 'Moderate','high': 'High', 'very active': 'High'}
    df.loc[:,'physical_activity'] = df['physical_activity'].str.strip().str.lower().map(activity_map)
    print("Physical activity column cleaned")
    
    df.loc[:,'dr_grade'] = df['dr_grade'].astype(str).str.replace('Grade ', '').str.strip()
    df.loc[:,'dr_grade'] = pd.to_numeric(df['dr_grade'], errors='coerce')
    print("DR grade column cleaned")
    
    df.loc[:,'years_diabetic'] = pd.to_numeric(df['years_diabetic'], errors='coerce')
    df.loc[df['years_diabetic'] < 0, 'years_diabetic'] = np.nan
    print("Years diabetic column cleaned")
    
    numeric_cols_to_clean = ['age', 'bmi', 'hba1c_level', 'systolic_bp', 'diastolic_bp', 'fasting_glucose', 'cholesterol']
    for col in numeric_cols_to_clean:
        df.loc[:,col] = pd.to_numeric(df[col], errors='coerce')

    print("Data cleaning complete.")
    print(df)
    return df
    
    
    

def main():
    raw_df = pd.read_csv('metadata.csv')
    print(f"Dataset loaded successfully.")
    
    df = clean_data(raw_df)
    
    
    
if __name__ == '__main__':
    main()