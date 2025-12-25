import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# MLflow Setup (WAJIB localhost)
# ===============================
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Hotel_Booking_Basic_Model")

def load_data():
    data = pd.read_csv("hotel_preprocessing/hotel_train.csv")
    X = data.drop("is_canceled", axis=1)
    y = data["is_canceled"]
    return X, y

def train_model():
    X, y = load_data()

    # WAJIB: train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    with mlflow.start_run():

        # WAJIB sesuai instruksi
        mlflow.sklearn.autolog()

        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Explicit metrics (aman walau autolog aktif)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

if __name__ == "__main__":
    train_model()