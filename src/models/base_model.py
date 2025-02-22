from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class BaseModel(ABC):
    """Abstract base class for all IDS models."""
    
    @abstractmethod
    def train(self, X: Any, y: Any) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Make predictions."""
        pass
    
    @abstractmethod
    def evaluate(self, X: Any, y: Any) -> Dict:
        """Evaluate model performance."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass