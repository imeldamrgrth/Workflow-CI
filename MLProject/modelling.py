import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log the model and accuracy with MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")

print(f"Model accuracy: {accuracy}")
