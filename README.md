# Aether-Data-Science-Forge

Aether-Data-Science-Forge is a professional, production-ready machine learning suite designed for scalability, reproducibility, and automated hyperparameter optimization.

## Architecture

The project follows a modular design to separate concerns between data processing, model tuning, and deployment.

- **Pipelines**: Automated training and validation pipelines using Scikit-learn and XGBoost.
- **Tuning**: Intelligent hyperparameter search powered by Optuna.
- **Serving**: High-performance model inference via FastAPI.
- **Ops**: Integrated with MLflow for experiment tracking and multi-stage Docker builds for containerized deployment.

## Project Structure

```
Aether-Data-Science-Forge/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ pipelines/
â”‚   â”‚   â””â”€â”€ training.py      # ML Pipeline logic
â”‚   â”œâ”€â”€ tuning/
â”‚   â”‚   â””â”€â”€ optimizer.py     # Optuna optimization
â”‚   â””â”€â”€ api/
â”‚       â””â”€â”€ inference.py    # FastAPI serving
â”œâ”€â”€ docker/
â”‚   â””â”€â”€ Dockerfile           # Multi-stage production build
â”œâ”€â”€ requirements.txt         # Dependency management
â””â”€â”€ README.md                # Documentation
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run Hyperparameter Tuning:
   ```bash
   python src/tuning/optimizer.py
   ```
3. Train the Model:
   ```bash
   python src/pipelines/training.py
   ```
4. Start Inference API:
   ```bash
   uvicorn src.api.inference:app --host 0.0.0.0 --port 8000
   ```

## MLOps Integration
- **Tracking**: Use MLflow to log parameters, metrics, and models.
- **Containerization**: Optimized Docker builds for lean production environments.
