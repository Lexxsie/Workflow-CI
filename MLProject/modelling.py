import os
import urllib.request
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
# CONFIG DATASET
# ======================================================
DATA_DIR = "data"
DATA_FILE = "hotel_train.csv"

# DIRECT DOWNLOAD LINK (WAJIB format ini)
DATA_URL = (
    "https://drive.google.com/uc?export=download&id="
    "1mFiv2KO_NQKSNcGfHilTfgkNk1Q72aDu"
)

# ======================================================
# DOWNLOAD DATASET (CI & LOCAL SAFE)
# ======================================================
def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    data_path = os.path.join(DATA_DIR, DATA_FILE)

    if not os.path.exists(data_path):
        print("Downloading dataset from Google Drive...")
        urllib.request.urlretrieve(DATA_URL, data_path)
        print("Dataset downloaded successfully")
    else:
        print("Dataset already exists")

    return data_path


# ======================================================
# LOAD & SPLIT DATA
# ======================================================
def load_data():
    data_path = download_dataset()

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if "is_canceled" not in df.columns:
        raise ValueError("Target column 'is_canceled' not found")

    X = df.drop("is_canceled", axis=1)
    y = df["is_canceled"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# ======================================================
# TRAIN MODEL
# ======================================================
def train():
    X_train, X_test, y_train, y_test = load_data()

    # ❗ PENTING
    # JANGAN set_experiment di MLflow Project
    # MLflow Project yang mengatur run & experiment
    with mlflow.start_run():
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # ------------------------------
        # MANUAL LOGGING (SKILLED READY)
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

        print("Training completed successfully")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    train()
