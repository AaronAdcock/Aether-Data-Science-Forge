import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import joblib
import os

def run_training_pipeline(data_path, params=None):
    # Default parameters if none provided
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8
        }

    mlflow.set_experiment("Aether-DS-Training")
    
    with mlflow.start_run():
        # Load data (using dummy data for demonstration)
        # In production: df = pd.read_csv(data_path)
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss'))
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Evaluate
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Model Accuracy: {acc:.4f}")
        print(classification_report(y_test, predictions))

        # Save model locally for API
        os.makedirs('models', exist_ok=True)
        joblib.dump(pipeline, 'models/model.joblib')

if __name__ == "__main__":
    run_training_pipeline(data_path=None)
