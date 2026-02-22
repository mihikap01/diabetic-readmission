"""Train SVM and XGBoost classifiers on the diabetic readmission dataset.

Based on the original test1.py: compares SVM (with SMOTE) against
XGBoost (with SMOTE) using RepeatedStratifiedKFold cross-validation
with F1 scoring.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

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
    print('diag fixed')

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
    print('data separated')

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
    pipeline_svm = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('over', SMOTE()),
        ('classifier', SVC(kernel='rbf', C=94.3553, gamma=13.3400))
    ])

    # Create a pipeline that combines the preprocessor with the XGBoost model
    pipeline_xgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('over', SMOTE()),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])

    # Evaluation function
    def evaluate_model(pipeline, X, y):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
        scores = cross_val_score(pipeline, X, y, scoring='f1', cv=cv, n_jobs=-1)
        return scores

    X = df[features]
    y = df[target]

    print('Evaluating SVM model...')
    scores_svm = evaluate_model(pipeline_svm, X, y)
    print(f'SVM F1 Score: {np.mean(scores_svm):.3f}')

    print('Evaluating XGBoost model...')
    scores_xgb = evaluate_model(pipeline_xgb, X, y)
    print(f'XGBoost F1 Score: {np.mean(scores_xgb):.3f}')
