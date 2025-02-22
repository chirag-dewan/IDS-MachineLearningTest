# src/models/random_forest.git adimport joblib
import numpy as np
from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from .base_model import BaseModel

class RandomForestIDS(BaseModel):
    """Random Forest implementation for IDS."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest model."""
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