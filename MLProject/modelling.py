import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# JANGAN set tracking_uri ke localhost di CI
mlflow.set_experiment("Hotel_Booking_Basic_Model")

data = pd.read_csv("namadataset_preprocessing/hotel_train.csv")
X = data.drop("is_canceled", axis=1)
y = data["is_canceled"]

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    acc = accuracy_score(y, model.predict(X))
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")
