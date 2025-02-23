import pytest
import numpy as np
import pandas as pd
from src.features.engineerer import FeatureEngineer

@pytest.fixture
def engineer():
    """Create a FeatureEngineer instance for testing."""
    return FeatureEngineer()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    data = np.random.rand(100, 5)
    columns = [f'feature_{i}' for i in range(5)]
    return pd.DataFrame(data, columns=columns)

def test_engineer_initialization(engineer):
    """Test engineer initialization."""
    assert engineer is not None
    assert not engineer._fitted

def test_statistical_features(engineer, sample_data):
    """Test statistical feature creation."""
    result = engineer._create_statistical_features(sample_data)
    
    # Check if statistical features were added
    assert 'mean' in result.columns
    assert 'std' in result.columns
    assert 'max' in result.columns
    assert 'min' in result.columns
    assert 'range' in result.columns
    assert 'mad' in result.columns
    
    # Check if original features are preserved
    for col in sample_data.columns:
        assert col in result.columns

def test_feature_scaling(engineer, sample_data):
    """Test feature scaling."""
    # Test fit_transform
    scaled_data = engineer._scale_features(sample_data, fit=True)
    assert scaled_data.shape == sample_data.shape
    
    # Check if data is scaled (mean ≈ 0, std ≈ 1)
    for col in scaled_data.columns:
        assert -0.1 < scaled_data[col].mean() < 0.1
        assert 0.9 < scaled_data[col].std() < 1.1

def test_fit_transform(engineer, sample_data):
    """Test fit_transform method."""
    result = engineer.fit_transform(sample_data)
    
    # Check if engineer is fitted
    assert engineer._fitted
    
    # Check if result has expected features
    expected_features = list(sample_data.columns) + ['mean', 'std', 'max', 'min', 'range', 'mad']
    assert all(feature in result.columns for feature in expected_features)

def test_transform_without_fit(engineer, sample_data):
    """Test transform without fitting first."""
    with pytest.raises(ValueError):
        engineer.transform(sample_data)

def test_transform_after_fit(engineer, sample_data):
    """Test transform after fitting."""
    # First fit_transform
    engineer.fit_transform(sample_data)
    
    # Then transform new data
    new_data = pd.DataFrame(np.random.rand(50, 5), columns=sample_data.columns)
    result = engineer.transform(new_data)
    
    # Check if result has expected features
    expected_features = list(sample_data.columns) + ['mean', 'std', 'max', 'min', 'range', 'mad']
    assert all(feature in result.columns for feature in expected_features)