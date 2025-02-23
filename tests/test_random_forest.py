import pytest
import numpy as np
import os
from src.models.random_forest import RandomForestIDS

@pytest.fixture
def model():
    """Create a RandomForestIDS instance for testing."""
    return RandomForestIDS(n_estimators=10, max_depth=5)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_model_initialization(model):
    """Test model initialization."""
    assert model is not None
    assert model.model.n_estimators == 10
    assert model.model.max_depth == 5

def test_model_training(model, sample_data):
    """Test model training."""
    X, y = sample_data
    model.train(X, y)
    assert model.model.n_features_in_ == X.shape[1]

def test_model_prediction(model, sample_data):
    """Test model prediction."""
    X, y = sample_data
    model.train(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)

def test_model_evaluation(model, sample_data):
    """Test model evaluation."""
    X, y = sample_data
    model.train(X, y)
    eval_results = model.evaluate(X, y)
    
    assert 'classification_report' in eval_results
    assert 'confusion_matrix' in eval_results
    assert isinstance(eval_results['confusion_matrix'], list)

def test_model_save_load(model, sample_data, tmp_path):
    """Test model saving and loading."""
    X, y = sample_data
    model.train(X, y)
    
    # Save model
    save_path = os.path.join(tmp_path, "model.joblib")
    model.save(save_path)
    assert os.path.exists(save_path)
    
    # Load model
    new_model = RandomForestIDS()
    new_model.load(save_path)
    
    # Compare predictions
    original_preds = model.predict(X)
    loaded_preds = new_model.predict(X)
    assert np.array_equal(original_preds, loaded_preds)