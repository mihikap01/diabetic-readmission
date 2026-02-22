"""Train an SVM classifier on the diabetic readmission dataset.

Based on the original diabetes-readmission.py: basic SVM with
StandardScaler + OneHotEncoder pipeline, no SMOTE.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from config import DATA_DIR
from preprocessing import load_data, fix_age, fix_weight, fix_diag


if __name__ == "__main__":
    # Load the data
    df = load_data()

    # Fix age and weight columns
    df['weight'] = df['weight'].apply(fix_weight)
    df['age'] = df['age'].apply(fix_age)

    # Fix diag_1 column
    df['diag_1'] = df['diag_1'].apply(fix_diag)
    df['diag_1'] = df['diag_1'].astype(float)
    df = df[(df['diag_1'] >= 250) & (df['diag_1'] <= 250.99)]
    df = df.dropna(subset=['race', 'diag_2', 'diag_3'])

    # Define features and target
    features = df.columns.difference(['readmitted'])
    target = 'readmitted'

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Create a pipeline that combines the preprocessor with the SVM model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC())
    ])

    # Split the data
    X = df[features]
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Predict on test data
    y_pred = pipeline.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
