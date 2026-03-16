"""Advanced Feature Engineering Module."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Automated feature engineering and selection."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.selector = None

    def transform(self, df: pd.DataFrame, target: str = None) -> Tuple[pd.DataFrame, List[str]]:
        """Applies scaling, encoding, and selection.

        Args:
            df: Input DataFrame.
            target: Name of the target column for feature selection.

        Returns:
            Transformed DataFrame and list of selected features.
        """
        logger.info("Starting automated feature engineering...")
        
        # Identify numerical and categorical columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if target and target in num_cols:
            num_cols.remove(target)
        if target and target in cat_cols:
            cat_cols.remove(target)

        # Scaling
        if num_cols:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
            logger.info(f"Scaled {len(num_cols)} numerical features.")

        # Encoding (Simplified for this example)
        # Note: In a production setting, you'd handle the sparse output properly
        
        # Feature Selection
        if target and target in df.columns:
            X = df.drop(columns=[target])
            y = df[target]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.selector = SelectFromModel(model, threshold="mean")
            self.selector.fit(X, y)
            selected_features = X.columns[self.selector.get_support()].tolist()
            logger.info(f"Selected {len(selected_features)} features based on importance.")
            return df[selected_features + [target]], selected_features

        return df, df.columns.tolist()
