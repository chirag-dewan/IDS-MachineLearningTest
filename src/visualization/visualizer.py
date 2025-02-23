import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
import numpy as np
from sklearn.metrics import roc_curve, auc

class IDSVisualizer:
    """Visualization tools for IDS analysis."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir
        plt.style.use('seaborn')
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance."""
        plt.figure(figsize=(10, 6))
        
        # Sort features by importance
        sorted_features = {
            k: v for k, v in sorted(
                feature_importance.items(),
                key=lambda item: item[1],
                reverse=True
            )
        }
        
        plt.bar(range(len(sorted_features)), list(sorted_features.values()))
        plt.xticks(
            range(len(sorted_features)),
            list(sorted_features.keys()),
            rotation=45,
            ha='right'
        )
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()