import joblib
import numpy as np
from typing import Any, Dict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from .base_model import BaseModel

class SVMIDS(BaseModel):
    """Support Vector Machine implementation for IDS."""
    
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: str = 'scale'):
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=42
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM model."""
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Returns:
            Dict containing classification report and confusion matrix
        """
        y_pred = self.predict(X)
        return {
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, path)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(path)