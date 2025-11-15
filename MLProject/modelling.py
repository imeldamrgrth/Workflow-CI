import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Muat data dan lakukan pra-pemrosesan
data = pd.read_csv("heart_processed.csv")  # Ganti dengan path dataset Anda
X = data.drop("target", axis=1)
y = data["target"]

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Prediksi dan hitung akurasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log model dengan MLflow
mlflow.start_run()
mlflow.log_param("model", "RandomForest")
mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()
