import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ======================================================
# MLflow Setup (CI-SAFE)
# ======================================================
# ❗ JANGAN set tracking_uri ke localhost di CI
# MLflow akan otomatis pakai file-based backend (./mlruns)

EXPERIMENT_NAME = "Hotel_Booking_Basic_Model"
mlflow.set_experiment(EXPERIMENT_NAME)


# ======================================================
# Load Dataset
# ======================================================
def load_data():
    data_path = "namadataset_preprocessing/hotel_train.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    X = df.drop("is_canceled", axis=1)
    y = df["is_canceled"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# ======================================================
# Train Model
# ======================================================
def train():
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run():
        # ------------------------------
        # Model
        # ------------------------------
        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        model.fit(X_train, y_train)

        # ------------------------------
        # Evaluation
        # ------------------------------
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # ------------------------------
        # MLflow Manual Logging
        # ------------------------------
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        # ------------------------------
        # Console Output (CI Friendly)
        # ------------------------------
        print("Training completed")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")


# ======================================================
# Entry Point
# ======================================================
if __name__ == "__main__":
    train()
