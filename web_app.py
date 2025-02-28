#!/usr/bin/env python3
"""
Web interface for the Intrusion Detection System (IDS).
Provides a user-friendly interface to run IDS analyses and view results.
"""

import os
import sys
import logging
import tempfile
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.config.config import Config
from src.features.engineerer import FeatureEngineer
from src.models.random_forest import RandomForestIDS
from src.models.svm import SVMIDS
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IDS.WebApp')

# Initialize Flask app
app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'ids_uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize global model and config
current_model = None
current_config = Config()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/config', methods=['GET', 'POST'])
def config():
    """Get or update configuration."""
    global current_config
    
    if request.method == 'POST':
        # Update configuration from form data
        config_data = request.json
        
        # Update config (would need proper validation in production)
        current_config._update_recursive(current_config.config, config_data)
        
        return jsonify({"status": "success"})
    
    # Return current configuration
    return jsonify(current_config.get())

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Update config with file path
        current_config.config['data']['source'] = 'csv'
        current_config.config['data']['path'] = filepath
        
        return jsonify({"status": "success", "filename": filename})
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/analyze', methods=['POST'])
def analyze():
    """Run IDS analysis."""
    global current_model
    
    try:
        # Get configuration
        data_config = current_config.get('data')
        model_config = current_config.get('model')
        output_config = current_config.get('output')
        
        # Create output directory
        output_dir = output_config.get('dir', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        if data_config.get('source') == 'synthetic':
            # Generate synthetic data (same as in ids.py)
            from sklearn.datasets import make_classification
            X_raw, y = make_classification(
                n_samples=10000, 
                n_features=20,
                n_informative=10,
                n_redundant=5,
                n_clusters_per_class=2,
                weights=[0.9, 0.1],
                random_state=42
            )
            feature_names = [f'feature_{i}' for i in range(20)]
            X = pd.DataFrame(X_raw, columns=feature_names)
        else:
            # Load CSV data
            data_path = data_config.get('path')
            if not data_path or not os.path.exists(data_path):
                return jsonify({"error": "Data file not found"}), 400
            
            data = pd.read_csv(data_path)
            target_column = data_config.get('target_column', 'label')
            
            if target_column not in data.columns:
                return jsonify({"error": f"Target column '{target_column}' not found"}), 400
                
            y = data[target_column].values
            X = data.drop(target_column, axis=1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=data_config.get('test_size', 0.2), random_state=42, stratify=y
        )
        
        # Feature engineering
        engineer = FeatureEngineer()
        X_train_eng = engineer.fit_transform(X_train)
        X_test_eng = engineer.transform(X_test)
        
        # Initialize model
        model_type = model_config.get('type', 'random_forest')
        model_params = model_config.get('params', {})
        
        if model_type == 'random_forest':
            current_model = RandomForestIDS(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 10)
            )
        elif model_type == 'svm':
            current_model = SVMIDS(
                C=model_params.get('C', 1.0),
                kernel=model_params.get('kernel', 'rbf'),
                gamma=model_params.get('gamma', 'scale')
            )
        else:
            return jsonify({"error": f"Unsupported model type: {model_type}"}), 400
        
        # Train model
        current_model.train(X_train_eng, y_train)
        
        # Save model if configured
        if model_config.get('save', False):
            model_path = os.path.join(output_dir, model_config.get('path', 'model.joblib'))
            current_model.save(model_path)
        
        # Evaluate model
        evaluation = current_model.evaluate(X_test_eng, y_test)
        
        # Generate visualizations
        from src.visualization.visualizer import IDSVisualizer
        visualizer = IDSVisualizer(output_dir=output_dir)
        
        # Confusion matrix
        y_pred = current_model.predict(X_test_eng)
        confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
        visualizer.plot_confusion_matrix(
            y_test, 
            y_pred,
            save_path=confusion_matrix_path
        )
        
        # ROC curve
        y_pred_proba = current_model.model.predict_proba(X_test_eng)[:, 1]
        roc_curve_path = os.path.join(output_dir, 'roc_curve.png')
        visualizer.plot_roc_curve(
            y_test, 
            y_pred_proba,
            save_path=roc_curve_path
        )
        
        # Feature importance (only for Random Forest)
        feature_importance_path = None
        if model_type == 'random_forest':
            feature_importance = dict(zip(
                X_train_eng.columns,
                current_model.model.feature_importances_
            ))
            feature_importance_path = os.path.join(output_dir, 'feature_importance.png')
            visualizer.plot_feature_importance(
                feature_importance,
                save_path=feature_importance_path
            )
        
        # Prepare response
        response = {
            "status": "success",
            "evaluation": evaluation,
            "visualizations": {
                "confusion_matrix": "/results/confusion_matrix.png",
                "roc_curve": "/results/roc_curve.png"
            }
        }
        
        if feature_importance_path:
            response["visualizations"]["feature_importance"] = "/results/feature_importance.png"
        
        return jsonify(response)
    
    except Exception as e:
        logger.exception("Error during analysis")
        return jsonify({"error": str(e)}), 500

@app.route('/results/<path:filename>')
def results(filename):
    """Serve result files."""
    output_dir = current_config.get('output').get('dir', 'output')
    return send_from_directory(output_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)