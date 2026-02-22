"""Train an MLP (Multi-Layer Perceptron) classifier on the diabetic readmission dataset.

Based on the original mlp-classifier-test.py: TensorFlow/Keras MLP with
two hidden layers (128 and 64 units), dropout regularization, and
softmax output for 3-class classification.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from config import DATA_DIR
from preprocessing import load_data, preprocess_data


if __name__ == "__main__":
    # Load the dataset
    data = load_data()

    # Data preprocessing
    X = data.drop(columns=["readmission"])
    y = data["readmission"]

    # Convert categorical labels to numerical labels
    y = pd.get_dummies(y)

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define the MLP model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 neurons for the three classes
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_scaled, y_train,
                        epochs=50,
                        batch_size=64,
                        validation_data=(X_val_scaled, y_val))

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(np.array(y_test), axis=1)

    # Classification report
    print(classification_report(y_true, y_pred_classes))
