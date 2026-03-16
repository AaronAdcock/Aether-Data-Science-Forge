"""Model Evaluation Suite."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Generates performance reports and visualizations."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def evaluate(self, y_true, y_score, model_name: str = "Model"):
        """Runs full evaluation suite.

        Args:
            y_true: Ground truth labels.
            y_score: Model probability scores.
            model_name: Name for the report.
        """
        logger.info(f"Evaluating {model_name}...")
        
        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f"{self.output_dir}/roc_auc_{model_name}.png")
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:0.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.savefig(f"{self.output_dir}/pr_curve_{model_name}.png")
        
        logger.info(f"Evaluation reports saved to {self.output_dir}")
