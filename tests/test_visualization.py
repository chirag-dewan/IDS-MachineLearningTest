import pytest
import numpy as np
import os
from src.visualization.visualizer import IDSVisualizer

@pytest.fixture
def visualizer():
    """Create an IDSVisualizer instance for testing."""
    return IDSVisualizer()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.rand(100)
    return y_true, y_pred, y_pred_proba

@pytest.fixture
def feature_importance_data():
    """Create sample feature importance data."""
    return {
        'feature_1': 0.3,
        'feature_2': 0.2,
        'feature_3': 0.5
    }

def test_visualizer_initialization(visualizer):
    """Test visualizer initialization."""
    assert visualizer is not None
    assert visualizer.output_dir is None

def test_confusion_matrix_plot(visualizer, sample_data, tmp_path):
    """Test confusion matrix plotting."""
    y_true, y_pred, _ = sample_data
    
    # Test saving plot
    save_path = os.path.join(tmp_path, "confusion_matrix.png")
    visualizer.plot_confusion_matrix(y_true, y_pred, save_path=save_path)
    assert os.path.exists(save_path)
    
    # Test showing plot (should not raise error)
    visualizer.plot_confusion_matrix(y_true, y_pred)

def test_roc_curve_plot(visualizer, sample_data, tmp_path):
    """Test ROC curve plotting."""
    y_true, _, y_pred_proba = sample_data
    
    # Test saving plot
    save_path = os.path.join(tmp_path, "roc_curve.png")
    visualizer.plot_roc_curve(y_true, y_pred_proba, save_path=save_path)
    assert os.path.exists(save_path)
    
    # Test showing plot (should not raise error)
    visualizer.plot_roc_curve(y_true, y_pred_proba)

def test_feature_importance_plot(visualizer, feature_importance_data, tmp_path):
    """Test feature importance plotting."""
    # Test saving plot
    save_path = os.path.join(tmp_path, "feature_importance.png")
    visualizer.plot_feature_importance(feature_importance_data, save_path=save_path)
    assert os.path.exists(save_path)
    
    # Test showing plot (should not raise error)
    visualizer.plot_feature_importance(feature_importance_data)