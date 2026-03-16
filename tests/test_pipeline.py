"""Integration tests for the ML lifecycle."""

import pytest
import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer

@pytest.fixture
def sample_data():
    """Generates sample data for testing."""
    return pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })

def test_data_loader(sample_data, tmp_path):
    """Tests the data loader validation."""
    csv_path = tmp_path / "test.csv"
    sample_data.to_csv(csv_path, index=False)
    
    loader = DataLoader()
    df = loader.load_csv(str(csv_path))
    assert len(df) == 100
    assert "target" in df.columns

def test_feature_engineering(sample_data):
    """Tests automated feature engineering."""
    engineer = FeatureEngineer()
    df_transformed, features = engineer.transform(sample_data, target='target')
    assert len(features) > 0
    assert 'target' in df_transformed.columns
