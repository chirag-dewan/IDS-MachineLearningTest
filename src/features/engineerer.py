import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class FeatureEngineer:
    """Handle feature engineering for IDS data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the feature engineering pipeline and transform the data.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        # Create statistical features
        X_engineered = self._create_statistical_features(X)
        
        # Scale features
        X_scaled = self._scale_features(X_engineered)
        
        self._fitted = True
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if not self._fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        X_engineered = self._create_statistical_features(X)
        return self._scale_features(X_engineered, fit=False)
    
    def _create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from raw data."""
        X_new = X.copy()
        
        # Add basic statistical features
        X_new['mean'] = X_new.mean(axis=1)
        X_new['std'] = X_new.std(axis=1)
        X_new['max'] = X_new.max(axis=1)
        X_new['min'] = X_new.min(axis=1)
        
        # Add more complex features
        X_new['range'] = X_new['max'] - X_new['min']
        X_new['mad'] = X_new.mad(axis=1)  # Mean absolute deviation
        
        return X_new
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        if fit:
            return pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns
            )
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns
        )