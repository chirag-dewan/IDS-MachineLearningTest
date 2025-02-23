import pytest
from src.models.base_model import BaseModel

def test_base_model_is_abstract():
    """Test that BaseModel cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseModel()

class ConcreteModel(BaseModel):
    """Concrete implementation for testing abstract methods."""
    def train(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
    def evaluate(self, X, y):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass

def test_concrete_model_creation():
    """Test that a concrete implementation can be instantiated."""
    model = ConcreteModel()
    assert isinstance(model, BaseModel)