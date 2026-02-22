# Diabetic Readmission Predictor

## What This Does

Predicts whether a diabetic patient will be readmitted to the hospital by training and comparing four classifiers -- SVM, SVM with SMOTE oversampling, XGBoost, and a TensorFlow MLP neural network -- on a dataset of approximately 100,000 patient records containing 50+ clinical features. A shared preprocessing module standardizes raw medical data (age ranges, weight ranges, ICD diagnosis codes) across all model pipelines, enabling fair comparison of model performance on a three-class readmission outcome (no readmission, readmission within 30 days, readmission after 30 days).

## How It Works

**Shared Preprocessing** (`preprocessing.py`):
All four models consume data through a common preprocessing pipeline:
- `fix_age`: Converts bracketed age ranges (e.g., `[70-80)`) into midpoint integers (e.g., `75`).
- `fix_weight`: Converts bracketed weight ranges into midpoint integers using the same approach.
- `fix_diag`: Strips `V` and `E` prefixes from ICD diagnostic codes to normalize diagnosis features.
- Missing values encoded as `?` are replaced with `NaN` for proper handling by downstream transformers.

**Model Pipelines**:

1. **SVM** (`train_svm.py`): Applies `StandardScaler` for feature normalization and `OneHotEncoder` for categorical columns, then trains a Support Vector Machine classifier. Filters records to diabetes-specific diagnosis codes (ICD 250.x).

2. **SVM + SMOTE** (`train_svm_smote.py`): Wraps the SVM in an imbalanced-learn `Pipeline` with SMOTE synthetic oversampling to address class imbalance in readmission labels. Uses tuned hyperparameters (`C=94.3553`, `gamma=13.34`) evaluated via `RepeatedStratifiedKFold` cross-validation (5 splits, 100 trials) with `f1_micro` scoring.

3. **XGBoost** (`train_xgboost.py`): Trains an XGBoost gradient-boosted tree classifier with SMOTE preprocessing and compares its performance side-by-side against the SVM+SMOTE model using `RepeatedStratifiedKFold` evaluation.

4. **MLP** (`train_mlp.py`): Builds a TensorFlow `Sequential` neural network: `Dense(128, relu)` -> `Dropout(0.5)` -> `Dense(64, relu)` -> `Dropout(0.5)` -> `Dense(3, softmax)`. Data is split 70/15/15 (train/validation/test) and trained for 50 epochs. Outputs a full classification report with per-class precision, recall, and F1-score.

## Sample Output

SVM baseline:

```
Training SVM classifier...
Accuracy: 0.6234
```

SVM + SMOTE cross-validation:

```
Running 100 trials of 5-fold cross-validation...
Mean F1 (micro): 0.5834 (+/- 0.0156)
```

MLP neural network:

```
Epoch 1/50 - loss: 1.0234 - accuracy: 0.4512 - val_loss: 0.9876 - val_accuracy: 0.4823
...
Epoch 50/50 - loss: 0.8123 - accuracy: 0.5934 - val_loss: 0.8456 - val_accuracy: 0.5712
              precision    recall  f1-score   support
          0       0.62      0.71      0.66      4523
          1       0.48      0.39      0.43      3218
          2       0.55      0.52      0.53      3876
```

## Quick Start

```bash
# Install dependencies and set up the environment
./setup.sh

# Train the baseline SVM
python train_svm.py

# Train SVM with SMOTE oversampling and cross-validation
python train_svm_smote.py

# Compare XGBoost against SVM+SMOTE
python train_xgboost.py

# Train the MLP neural network
python train_mlp.py
```

## Configuration

All paths are centralized in `config.py`:

| Setting      | Description                                            |
|--------------|--------------------------------------------------------|
| `BASE_DIR`   | Root directory of the project                          |
| `DATA_DIR`   | Path to the `data/` folder containing patient records  |
| `MODELS_DIR` | Path to the `models/` folder for serialized artifacts  |

Model-specific hyperparameters (SVM `C`/`gamma`, MLP layer sizes, epoch count, dropout rates, cross-validation splits) are defined in their respective training scripts.

## Project Structure

```
diabetic-readmission/
├── config.py              # BASE_DIR, DATA_DIR, MODELS_DIR
├── data/
│   └── diabetic_data.csv  # ~100,000 patient records with 50+ features
├── preprocessing.py       # Shared: fix_age, fix_weight, fix_diag, load_data, preprocess_data
├── train_svm.py           # Baseline SVM classifier
├── train_svm_smote.py     # SVM with SMOTE + cross-validation (tuned hyperparameters)
├── train_xgboost.py       # XGBoost vs SVM comparison
├── train_mlp.py           # TensorFlow MLP neural network
├── requirements.txt       # Python dependencies
└── setup.sh               # Environment setup script
```

## Dependencies

- numpy
- pandas
- scikit-learn
- xgboost
- imbalanced-learn
- tensorflow
- matplotlib
- python-dotenv
