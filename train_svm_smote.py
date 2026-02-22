"""Train an SVM classifier with SMOTE on the diabetic readmission dataset.

Based on the original test.py: SVM with SMOTE oversampling, tuned
hyperparameters (C=94.3553, gamma=13.3400), cross-validated with
RepeatedStratifiedKFold over 100 trials.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import accuracy_score

from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from config import DATA_DIR
from preprocessing import load_data, fix_age, fix_weight, fix_diag


if __name__ == "__main__":
    print('imported libraries')

    # Load the data
    df = load_data()
    print("data loaded")

    # Fix age column
    df['age'] = df['age'].apply(fix_age)
    print('age fixed')

    df['medical_specialty'] = df['medical_specialty'].fillna('missing')

    # Fix diag_1 column
    df['diag_1'] = df['diag_1'].apply(fix_diag)
    df['diag_1'] = df['diag_1'].astype(float)
    df = df[(df['diag_1'] >= 250) & (df['diag_1'] <= 250.99)]
    df = df.dropna(subset=['race', 'diag_2', 'diag_3'])
    print('diag')

    # Fix weight column
    df['weight'] = df['weight'].apply(fix_weight)
    print('weight fixed')

    a1c_mapping = {'>7': 1, '>8': 0, 'Norm': 2, np.nan: 3}
    df['A1Cresult'] = df['A1Cresult'].map(a1c_mapping)

    # Encode the readmitted column
    readmitted_mapping = {'<30': 1, '>30': 0, 'NO': 0}
    df['readmitted'] = df['readmitted'].map(readmitted_mapping)

    # Ensure that 'readmitted' column is correctly mapped
    if df['readmitted'].isnull().any():
        raise ValueError("There are unmapped values in the 'readmitted' column.")
    print('classes mapped')

    bad_features = ['chlorpropamide', 'acetohexamide', 'tolbutamide', 'miglitol',
                    'troglitazone', 'tolazamide', 'examide', 'citoglipton',
                    'glipizide-metformin', 'glimepiride-pioglitazone',
                    'metformin-rosiglitazone', 'metformin-pioglitazone',
                    'readmitted', 'diag_2', 'diag_3',
                    'encounter_id', 'patient_nbr', 'payer_code', 'weight']

    # Define features and target
    features = df.columns.difference(bad_features)
    target = 'readmitted'

    # Identify numerical and categorical columns
    numerical_cols = df[features].select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df[features].select_dtypes(include=['object']).columns
    print('data seperated')

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SklearnPipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', SklearnPipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ])

    # Create a pipeline that combines the preprocessor with the SVM model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('over', SMOTE()),
        ('classifier', SVC(kernel='rbf', C=94.3553, gamma=13.3400))
    ])

    print("Half of code ran, just f1 score left")

    f1s = []
    for trial in range(100):
        #cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
        scores = cross_val_score(pipeline, df[features], df[target], scoring='f1_micro', cv=cv, n_jobs=-1)
        score = np.mean(scores)
        f1s.append(score)
        print(score)

    print(f1s)
