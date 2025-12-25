import os
import mlflow
import mlflow.sklearn
import pandas as pd
import gdown
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# CONFIG
# =====================================================
EXPERIMENT_NAME = "Hotel_Booking_Basic_Model"
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "hotel_train.csv")
GDRIVE_FILE_ID = "1mFiv2KO_NQKSNcGfHilTfgkNk1Q72aDu"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# =====================================================
# UTILS
# =====================================================
def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset from Google Drive...")
        gdown.download(GDRIVE_URL, DATA_PATH, quiet=False)
        print("Dataset downloaded successfully")
    else:
        print("Dataset already exists")

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    X = df.drop("is_canceled", axis=1)
    y = df["is_canceled"]
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

# =====================================================
# TRAINING (MLFLOW PROJECT SAFE)
# =====================================================
def train():
    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Check jika ada run aktif dari CLI
    active_run = mlflow.active_run()
    
    if active_run is None:
        # Tidak ada run aktif, buat baru
        with mlflow.start_run():
            _train_model()
    else:
        # Sudah ada run aktif dari CLI, gunakan itu
        _train_model()

def _train_model():
    """Helper function untuk training logic"""
    X_train, X_test, y_train, y_test = load_data()
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # Log parameters dan metrics
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )
    
    print(f"Training finished | Accuracy = {acc:.4f}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    download_dataset()
    train()
