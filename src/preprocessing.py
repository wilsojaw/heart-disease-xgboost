import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # One hot encode
    df = pd.get_dummies(df, columns=['ChestPainType'], drop_first=True)
    df = pd.get_dummies(df, columns=['RestingECG'], drop_first=True)
    df = pd.get_dummies(df, columns=['ST_Slope'], drop_first=True)

    df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
    df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})

    # AgeGroup: Young < Middle < Old
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 40, 60, 120], labels=['Young', 'Middle', 'Old'])
    df['AgeGroup'] = pd.Categorical(df['AgeGroup'], categories=['Young', 'Middle', 'Old'], ordered=True)
    df['AgeGroup_encoded'] = df['AgeGroup'].cat.codes

    # BPGroup: Normal < Prehypertension < Hypertension
    df['BPGroup'] = pd.cut(df['RestingBP'], bins=[0, 120, 140, 200], labels=['Normal', 'Prehypertension', 'Hypertension'])
    df['BPGroup'] = pd.Categorical(df['BPGroup'], categories=['Normal', 'Prehypertension', 'Hypertension'], ordered=True)
    df['BPGroup_encoded'] = df['BPGroup'].cat.codes

    # CholGroup: Normal < Borderline < High
    df['CholGroup'] = pd.cut(df['Cholesterol'], bins=[0, 200, 240, 600], labels=['Normal', 'Borderline', 'High'])
    df['CholGroup'] = pd.Categorical(df['CholGroup'], categories=['Normal', 'Borderline', 'High'], ordered=True)
    df['CholGroup_encoded'] = df['CholGroup'].cat.codes

    # OldpeakGroup: Normal < Mild Depression < Severe Depression
    df['OldpeakGroup'] = pd.cut(df['Oldpeak'], bins=[-1, 0, 2, 10], labels=['Normal', 'Mild Depression', 'Severe Depression'])
    df['OldpeakGroup'] = pd.Categorical(df['OldpeakGroup'], categories=['Normal', 'Mild Depression', 'Severe Depression'], ordered=True)
    df['OldpeakGroup_encoded'] = df['OldpeakGroup'].cat.codes

    # Drop original binned group columns
    df = df.drop(columns=['AgeGroup', 'BPGroup', 'CholGroup', 'OldpeakGroup'])

    # Optional HRGroup check
    if 'HRGroup' in df.columns:
        df['HRGroup_encoded'] = df['HRGroup'].cat.codes
        df = df.drop(columns=['HRGroup'])

    # Ensure all expected dummy columns from training are present
    expected_dummies = [
        'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_Normal', 'RestingECG_ST', 
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    for col in expected_dummies:
        if col not in df.columns:
            df[col] = 0

    # Optional: reorder for consistency
    df = df.sort_index(axis=1)

    return df

def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor