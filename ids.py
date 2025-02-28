#!/usr/bin/env python3
"""
Intrusion Detection System (IDS) Main Script

This script demonstrates a machine learning-based intrusion detection system.
It uses synthetic data by default but can be configured to use real datasets.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.config.config import Config

from src.features.engineerer import FeatureEngineer
from src.models.random_forest import RandomForestIDS
from src.models.svm import SVMIDS
from src.visualization.visualizer import IDSVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IDS')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Machine Learning-based IDS')
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        choices=['synthetic', 'csv'],
        help='Data source (synthetic or csv)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to CSV dataset when using --data=csv'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        help='Test data size (proportion)'
    )
    
    parser.add_argument(
        '--n-estimators',
        type=int,
        help='Number of trees in Random Forest'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        help='Maximum depth of trees in Random Forest'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory for output files'
    )
    
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save the trained model'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to save or load model'
    )
    
    parser.add_argument(
        '--load-model',
        action='store_true',
        help='Load model instead of training'
    )
    
    parser.add_argument(
        '--save-config',
        type=str,
        help='Save current configuration to specified path'
    )
    
    return parser.parse_args()

def generate_synthetic_data(n_samples=10000, n_features=20):
    """Generate synthetic data for IDS demonstration."""
    logger.info(f"Generating synthetic dataset with {n_samples} samples")
    
    # Create imbalanced dataset (90% benign, 10% malicious)
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    # Convert to DataFrame for better handling
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df, y

def load_csv_data(data_path, target_column='label'):
    """Load data from CSV file."""
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    try:
        data = pd.read_csv(data_path)
        if target_column not in data.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            sys.exit(1)
            
        y = data[target_column].values
        X = data.drop(target_column, axis=1)
        
        return X, y
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)

def prepare_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load configuration
    config = Config(args.config if hasattr(args, 'config') and args.config else None)
    
    # Update config with command line arguments
    config.update_from_args(vars(args))
    
    # Save config if requested
    if hasattr(args, 'save_config') and args.save_config:
        config.save_to_file(args.save_config)
    
    # Get configuration sections
    data_config = config.get('data')
    model_config = config.get('model')
    output_config = config.get('output')
    feature_config = config.get('features')
    
    # Prepare output directory
    output_dir = output_config.get('dir', 'output')
    prepare_output_directory(output_dir)
    
    # Load or generate data
    if data_config.get('source') == 'synthetic':
        X, y = generate_synthetic_data()
    else:  # data_config.get('source') == 'csv'
        data_path = data_config.get('path')
        if not data_path:
            logger.error("Data path is required when using CSV data source")
            sys.exit(1)
        X, y = load_csv_data(data_path, target_column=data_config.get('target_column', 'label'))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_config.get('test_size', 0.2), random_state=42, stratify=y
    )
    
    logger.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Feature engineering
    logger.info("Performing feature engineering")
    engineer = FeatureEngineer()
    X_train_eng = engineer.fit_transform(X_train)
    X_test_eng = engineer.transform(X_test)
    
    # Initialize model based on configuration
    model_type = model_config.get('type', 'random_forest')
    model_params = model_config.get('params', {})
    
    # Select model based on configuration
    if model_type == 'random_forest':
        model = RandomForestIDS(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10)
        )
    elif model_type == 'svm':
        model = SVMIDS(
            C=model_params.get('C', 1.0),
            kernel=model_params.get('kernel', 'rbf'),
            gamma=model_params.get('gamma', 'scale')
        )
    else:
        logger.error(f"Unsupported model type: {model_type}")
        sys.exit(1)
    
    # Train or load model
    load_model = args.load_model if hasattr(args, 'load_model') else False
    if load_model:
        model_path = model_config.get('path', 'model.joblib')
        logger.info(f"Loading model from {model_path}")
        model.load(model_path)
    else:
        logger.info("Training model")
        model.train(X_train_eng, y_train)
        
        if model_config.get('save', False):
            model_path = os.path.join(output_dir, model_config.get('path', 'model.joblib'))
            logger.info(f"Saving model to {model_path}")
            model.save(model_path)
    
    # Evaluate model
    logger.info("Evaluating model")
    evaluation = model.evaluate(X_test_eng, y_test)
    
    # Print classification report
    logger.info("Classification Report:")
    for label, metrics in evaluation['classification_report'].items():
        if isinstance(metrics, dict):
            logger.info(f"  {label}:")
            for metric_name, value in metrics.items():
                logger.info(f"    {metric_name}: {value:.4f}")
    
    # Save visualizations if configured
    if output_config.get('save_plots', True):
        # Get feature importance
        feature_importance = dict(zip(
            X_train_eng.columns,
            model.model.feature_importances_
        ))
        
        # Visualize results
        logger.info("Generating visualizations")
        visualizer = IDSVisualizer(output_dir=output_dir)
        
        # Confusion matrix
        y_pred = model.predict(X_test_eng)
        visualizer.plot_confusion_matrix(
            y_test, 
            y_pred,
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
        
        # ROC curve
        y_pred_proba = model.model.predict_proba(X_test_eng)[:, 1]
        visualizer.plot_roc_curve(
            y_test, 
            y_pred_proba,
            save_path=os.path.join(output_dir, 'roc_curve.png')
        )
        
        # Feature importance
        visualizer.plot_feature_importance(
            feature_importance,
            save_path=os.path.join(output_dir, 'feature_importance.png')
        )
        
        logger.info(f"Results saved to {output_dir}")
    
    logger.info("IDS analysis complete")

if __name__ == "__main__":
    main()