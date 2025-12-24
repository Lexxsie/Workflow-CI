import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data():
    X_train = np.load("namadataset_preprocessing/X_train.npy")
    X_test = np.load("namadataset_preprocessing/X_test.npy")
    y_train = np.load("namadataset_preprocessing/y_train.npy")
    y_test = np.load("namadataset_preprocessing/y_test.npy")
    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_data()

    # WAJIB: file-based tracking (CI & local serving)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CI_Retraining_Model")

    # Autolog untuk parameter & metric (BOLEH)
    mlflow.sklearn.autolog(log_models=False)

    # ⚠️ JANGAN pakai mlflow.start_run() di MLflow Project
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Inference
    y_pred = model.predict(X_test)

    # Metrics (tetap dicetak untuk log CI)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    # 🔥 INI BAGIAN PALING PENTING
    # WAJIB agar artifacts/model terbentuk
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )


if __name__ == "__main__":
    main()