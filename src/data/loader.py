"""Data Loading and Validation Module."""

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSchema(BaseModel):
    """Pydantic schema for data validation."""
    feature_1: float = Field(..., description="A key numerical feature")
    feature_2: float = Field(..., description="Another numerical feature")
    target: int = Field(..., ge=0, le=1, description="Binary classification target")

class DataLoader:
    """Robust data loader with validation."""

    def __init__(self, schema_class: BaseModel = DataSchema):
        self.schema_class = schema_class

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Loads and validates a CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Validated pandas DataFrame.

        Raises:
            ValueError: If data validation fails.
        """
        try:
            df = pd.read_csv(file_path)
            self.validate_data(df)
            logger.info(f"Successfully loaded and validated {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def validate_data(self, df: pd.DataFrame):
        """Validates DataFrame against the Pydantic schema.

        Args:
            df: The DataFrame to validate.
        """
        for _, row in df.iterrows():
            try:
                self.schema_class(**row.to_dict())
            except ValidationError as e:
                logger.error(f"Row validation failed: {e}")
                raise ValueError(f"Invalid data format: {e}")
