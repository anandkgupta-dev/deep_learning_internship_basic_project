"""
train_model.py
==============
Beginner-Friendly Deep Learning Training Script for Weather Condition Classification.

This script:
1. Loads the synthetic weather dataset (temperature and humidity).
2. Performs data preprocessing (encoding, stratified splitting, StandardScaler).
3. Constructs a 2-layer Neural Network (Input -> 8 Hidden Neurons ReLU -> 3 Output Neurons Softmax).
4. Compiles with Adam optimizer and Sparse Categorical Cross-Entropy loss.
5. Trains for 50 epochs and captures history.
6. Generates training/validation loss and accuracy learning curves.
7. Evaluates performance on the held-out test dataset (Loss, Accuracy, Precision, Recall, F1, Confusion Matrix).
8. Saves all artifacts: model (.keras), scaler (.joblib), plots (.png), and metrics (.json).
"""

import os
import json
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# TensorFlow imports
import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow import keras
from tensorflow.keras import layers

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


def main():
    print("=" * 70)
    print("  WEATHER CONDITION CLASSIFICATION - MODEL TRAINING PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 1: Ensure directories and dataset exist
    # ---------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "weather_data.csv")
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Running generate_data.py first...")
        from generate_data import generate_weather_dataset
        df = generate_weather_dataset(num_samples=1000, random_seed=SEED)
        df.to_csv(csv_path, index=False)
        print("Dataset generated successfully.")

    # ---------------------------------------------------------
    # Step 2: Load and Inspect Dataset
    # ---------------------------------------------------------
    print("\n[Step 1/8] Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nSample rows (first 5):")
    print(df.head())

    print("\nTarget Class Distribution:")
    print(df["weather_condition"].value_counts())

    # ---------------------------------------------------------
    # Step 3: Feature & Label Separation and Encoding
    # ---------------------------------------------------------
    print("\n[Step 2/8] Separating features (X) and target labels (y)...")
    # Features: Temperature (°C) and Relative Humidity (%)
    feature_names = ["temperature_c", "humidity_pct"]
    X = df[feature_names].values

    # Target classes: Sunny (0), Cloudy (1), Rainy (2)
    class_names = ["Sunny", "Cloudy", "Rainy"]
    label_to_index = {name: idx for idx, name in enumerate(class_names)}
    index_to_label = {idx: name for idx, name in enumerate(class_names)}

    y = df["weather_condition"].map(label_to_index).values
    print(f"Class mapping: {label_to_index}")

    # ---------------------------------------------------------
    # Step 4: Train / Test Split (Stratified to maintain class balance)
    # ---------------------------------------------------------
    print("\n[Step 3/8] Splitting data into 80% Training and 20% Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y  # Ensures equal proportion of Sunny/Cloudy/Rainy in train and test
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # ---------------------------------------------------------
    # Step 5: Feature Scaling using StandardScaler
    # ---------------------------------------------------------
    print("\n[Step 4/8] Applying StandardScaler to numerical features...")
    # Why? Temperature (5-45) and Humidity (15-100) are on different scales.
    # StandardScaler transforms features to mean=0 and variance=1 so gradient descent converges smoothly.
    # CRITICAL: Fit ONLY on training set to avoid data leakage!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save fitted scaler for use in the Streamlit web application
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted StandardScaler to: {scaler_path}")

    # ---------------------------------------------------------
    # Step 6: Build Basic Feed-Forward Neural Network
    # ---------------------------------------------------------
    print("\n[Step 5/8] Constructing Neural Network Architecture...")
    # Architecture explanation:
    # 1. Input Layer: 2 neurons (Feature 1: Temperature, Feature 2: Humidity)
    # 2. Hidden Layer: 8 neurons with ReLU activation (allows network to learn non-linear decision boundaries)
    # 3. Output Layer: 3 neurons with Softmax activation (outputs probability distribution across the 3 classes)
    model = keras.Sequential([
        layers.Input(shape=(2,), name="input_layer"),
        layers.Dense(8, activation="relu", name="hidden_layer_relu"),
        layers.Dense(3, activation="softmax", name="output_layer_softmax")
    ], name="Weather_Classifier_ANN")

    model.summary()

    # Compile the model
    # - Optimizer: Adam (adaptive learning rate for steady convergence)
    # - Loss: Sparse Categorical Cross-Entropy (used when true labels are integers: 0, 1, 2)
    # - Metric: Accuracy (percentage of correct classifications)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ---------------------------------------------------------
    # Step 7: Train the Model
    # ---------------------------------------------------------
    print("\n[Step 6/8] Training model for 50 epochs...")
    EPOCHS = 50
    BATCH_SIZE = 16

    # We use validation_split=0.2 from training set to monitor validation loss during learning
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.20,
        verbose=1
    )

    # ---------------------------------------------------------
    # Step 8: Plot and Save Learning Curves (Loss & Accuracy)
    # ---------------------------------------------------------
    print("\n[Step 7/8] Generating and saving training curves...")
    epochs_range = range(1, EPOCHS + 1)
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plt.figure(figsize=(12, 5))

    # Plot 1: Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label="Training Loss", color="#e74c3c", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#c0392b", linestyle="--", linewidth=2)
    plt.title("Model Loss vs. Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=11)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Plot 2: Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label="Training Accuracy", color="#2ecc71", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#27ae60", linestyle="--", linewidth=2)
    plt.title("Model Accuracy vs. Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    history_plot_path = os.path.join(models_dir, "training_history.png")
    plt.savefig(history_plot_path, dpi=200)
    plt.close()
    print(f"Saved training history curves to: {history_plot_path}")

    # ---------------------------------------------------------
    # Step 9: Evaluate on Held-Out Test Dataset
    # ---------------------------------------------------------
    print("\n[Step 8/8] Evaluating model on unseen test set...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"-> Test Loss:     {test_loss:.4f}")
    print(f"-> Test Accuracy: {test_accuracy * 100:.2f}%")

    # Predict probabilities on test set
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification Report
    cr_dict = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True
    )
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot Confusion Matrix using Matplotlib
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Test Set Confusion Matrix", fontsize=13, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    # Annotate counts inside the confusion matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight="bold"
            )

    plt.ylabel("True Actual Label", fontsize=11)
    plt.xlabel("Model Predicted Label", fontsize=11)
    plt.tight_layout()
    cm_plot_path = os.path.join(models_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    # Save Model in Keras native format (.keras)
    model_path = os.path.join(models_dir, "weather_model.keras")
    model.save(model_path)
    print(f"Saved trained neural network model to: {model_path}")

    # Save complete evaluation metrics to JSON
    eval_results = {
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "test_accuracy_pct": round(float(test_accuracy * 100), 2),
        "total_test_samples": len(y_test),
        "class_names": class_names,
        "classification_report": cr_dict,
        "confusion_matrix": cm.tolist(),
        "training_epochs": EPOCHS,
        "final_train_loss": round(float(train_loss[-1]), 4),
        "final_train_accuracy": round(float(train_acc[-1]), 4),
        "final_val_loss": round(float(val_loss[-1]), 4),
        "final_val_accuracy": round(float(val_acc[-1]), 4)
    }

    eval_json_path = os.path.join(models_dir, "evaluation_results.json")
    with open(eval_json_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Saved detailed evaluation metrics to: {eval_json_path}")

    print("\n" + "=" * 70)
    print("  TRAINING & EVALUATION COMPLETE!")
    print(f"  Final Test Accuracy: {test_accuracy * 100:.2f}% | Test Loss: {test_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
