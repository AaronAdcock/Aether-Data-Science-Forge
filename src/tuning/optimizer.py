import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import mlflow

def objective(trial):
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    }

    # Load data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    clf = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    
    # Use cross-validation for robustness
    score = cross_val_score(clf, X, y, n_jobs=-1, cv=3).mean()
    
    return score

def run_optimization():
    mlflow.set_experiment("Aether-DS-Optimization")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Log best params to MLflow
    with mlflow.start_run(run_name="Best_Trial"):
        mlflow.log_params(trial.params)
        mlflow.log_metric("best_accuracy", trial.value)

if __name__ == "__main__":
    run_optimization()
