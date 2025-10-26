"""
Model Performance Visualization Module
Provides functions for visualizing and comparing machine learning model results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import logging


class ModelVisualizer:
    """
    Handles visualization of model performance metrics and comparisons.
    Creates plots for confusion matrices, ROC curves, and metric comparisons.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize model visualizer with plotting style.
        
        Args:
            style: Matplotlib style to use for plots
        """
        self.logger = logging.getLogger(__name__)
        plt.style.use(style)
        
        # Color scheme for models
        self.model_colors = {
            'Naive Bayes': '#3498db',        # Blue
            'Logistic Regression': '#e74c3c', # Red
            'Random Forest': '#2ecc71'        # Green
        }
    
    def plot_confusion_matrices(self, results: Dict, save_path: str = None) -> plt.Figure:
        """
        Create a grid of confusion matrices for all models.
        
        Args:
            results: Dictionary mapping model names to ModelResult objects
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(results.items()):
            ax = axes[idx]
            cm = result.confusion_matrix
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       cbar=False, ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       annot_kws={'size': 14, 'weight': 'bold'})
            
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name}\nAccuracy: {result.accuracy:.4f}',
                        fontsize=12, fontweight='bold', pad=15)
            
            # Add percentage annotations
            total = cm.sum()
            for i in range(2):
                for j in range(2):
                    percentage = (cm[i, j] / total) * 100
                    ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                           ha='center', va='center', fontsize=9, color='gray')
        
        plt.suptitle('Confusion Matrices - Model Comparison', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrices saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, models: Dict, X_test: np.ndarray, 
                       y_test: np.ndarray, save_path: str = None) -> plt.Figure:
        """
        Plot ROC curves for all models on the same figure.
        
        Args:
            models: Dictionary mapping model names to trained model objects
            X_test: Test features (vectorized)
            y_test: Test labels (encoded as 0/1)
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, model in models.items():
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                y_scores = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_scores = model.decision_function(X_test)
            else:
                self.logger.warning(f"{model_name} doesn't support probability estimates")
                continue
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, 
                   color=self.model_colors.get(model_name, 'gray'),
                   lw=2.5, alpha=0.8,
                   label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Model Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curves saved to {save_path}")
        
        return fig
    
    def plot_metrics_comparison(self, results: Dict, save_path: str = None) -> plt.Figure:
        """
        Create grouped bar chart comparing metrics across all models.
        
        Args:
            results: Dictionary mapping model names to ModelResult objects
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Prepare data
        models = list(results.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        data = {
            'Accuracy': [results[m].accuracy for m in models],
            'Precision': [results[m].precision for m in models],
            'Recall': [results[m].recall for m in models],
            'F1-Score': [results[m].f1_score for m in models]
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.arange(len(models))
        width = 0.2
        multiplier = 0
        
        for metric, values in data.items():
            offset = width * multiplier
            bars = ax.bar(x + offset, values, width, label=metric, alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            multiplier += 1
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Metrics comparison saved to {save_path}")
        
        return fig
    
    def plot_training_times(self, results: Dict, save_path: str = None) -> plt.Figure:
        """
        Create bar chart showing training and prediction times.
        
        Args:
            results: Dictionary mapping model names to ModelResult objects
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        models = list(results.keys())
        train_times = [results[m].training_time for m in models]
        pred_times = [results[m].prediction_time * 1000 for m in models]  # Convert to ms
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Training time
        colors_train = [self.model_colors.get(m, 'gray') for m in models]
        bars1 = ax1.bar(models, train_times, color=colors_train, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax1.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Prediction time
        bars2 = ax2.bar(models, pred_times, color=colors_train, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Time (milliseconds)', fontsize=11, fontweight='bold')
        ax2.set_title('Prediction Time Comparison', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}ms',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Model Efficiency Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training times plot saved to {save_path}")
        
        return fig
    
    def create_comparison_table_styled(self, results: Dict) -> pd.DataFrame:
        """
        Create a styled comparison table of model performance.
        
        Args:
            results: Dictionary mapping model names to ModelResult objects
            
        Returns:
            Styled pandas DataFrame
        """
        data = []
        for name, result in results.items():
            data.append({
                'Model': name,
                'Accuracy': result.accuracy,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1-Score': result.f1_score,
                'Training Time (s)': result.training_time,
                'Prediction Time (ms)': result.prediction_time * 1000
            })
        
        df = pd.DataFrame(data)
        
        # Sort by F1-Score descending
        df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
        
        # Apply styling
        styled_df = df.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}',
            'Training Time (s)': '{:.2f}',
            'Prediction Time (ms)': '{:.2f}'
        }).background_gradient(
            subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            cmap='Greens',
            vmin=0.7,
            vmax=1.0
        ).highlight_max(
            subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            color='lightgreen'
        ).highlight_min(
            subset=['Training Time (s)', 'Prediction Time (ms)'],
            color='lightblue'
        ).set_properties(**{
            'text-align': 'center',
            'font-size': '11pt'
        }).set_table_styles([
            {'selector': 'th', 'props': [('font-weight', 'bold'), 
                                         ('text-align', 'center'),
                                         ('background-color', '#f0f0f0')]}
        ])
        
        return styled_df
    
    def create_model_dashboard(self, results: Dict, models: Dict,
                              X_test: np.ndarray, y_test: np.ndarray,
                              save_path: str = None) -> plt.Figure:
        """
        Create comprehensive dashboard with all model visualizations.
        
        Args:
            results: Dictionary mapping model names to ModelResult objects
            models: Dictionary mapping model names to trained model objects
            X_test: Test features (vectorized)
            y_test: Test labels (encoded as 0/1)
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrices (top row)
        n_models = len(results)
        for idx, (model_name, result) in enumerate(results.items()):
            ax = fig.add_subplot(gs[0, idx])
            cm = result.confusion_matrix
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       cbar=False, ax=ax,
                       xticklabels=['Neg', 'Pos'],
                       yticklabels=['Neg', 'Pos'],
                       annot_kws={'size': 12, 'weight': 'bold'})
            
            ax.set_title(f'{model_name}\nAcc: {result.accuracy:.3f}',
                        fontsize=11, fontweight='bold')
        
        # 2. ROC Curves (middle left)
        ax_roc = fig.add_subplot(gs[1, :2])
        for model_name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_scores = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, lw=2.5, alpha=0.8,
                           color=self.model_colors.get(model_name, 'gray'),
                           label=f'{model_name} (AUC={roc_auc:.3f})')
        
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
        ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
        ax_roc.set_title('ROC Curves', fontweight='bold')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(alpha=0.3)
        
        # 3. Metrics Comparison (middle right)
        ax_metrics = fig.add_subplot(gs[1, 2])
        model_names = list(results.keys())
        metrics_data = {
            'Accuracy': [results[m].accuracy for m in model_names],
            'Precision': [results[m].precision for m in model_names],
            'Recall': [results[m].recall for m in model_names],
            'F1': [results[m].f1_score for m in model_names]
        }
        
        x = np.arange(len(model_names))
        width = 0.2
        for i, (metric, values) in enumerate(metrics_data.items()):
            ax_metrics.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax_metrics.set_xticks(x + width * 1.5)
        ax_metrics.set_xticklabels([m.split()[0] for m in model_names], 
                                   fontsize=9, rotation=15)
        ax_metrics.set_ylabel('Score', fontweight='bold')
        ax_metrics.set_title('Performance Metrics', fontweight='bold')
        ax_metrics.legend(fontsize=8)
        ax_metrics.set_ylim([0, 1.1])
        ax_metrics.grid(axis='y', alpha=0.3)
        
        # 4. Training Times (bottom left)
        ax_train = fig.add_subplot(gs[2, 0])
        train_times = [results[m].training_time for m in model_names]
        colors = [self.model_colors.get(m, 'gray') for m in model_names]
        bars = ax_train.bar(range(len(model_names)), train_times, 
                           color=colors, alpha=0.8, edgecolor='black')
        ax_train.set_xticks(range(len(model_names)))
        ax_train.set_xticklabels([m.split()[0] for m in model_names], 
                                 fontsize=9, rotation=15)
        ax_train.set_ylabel('Seconds', fontweight='bold')
        ax_train.set_title('Training Time', fontweight='bold')
        ax_train.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax_train.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
        
        # 5. Performance Summary Table (bottom middle)
        ax_table = fig.add_subplot(gs[2, 1])
        ax_table.axis('off')
        
        table_data = []
        for name, result in results.items():
            table_data.append([
                name.split()[0],
                f'{result.accuracy:.3f}',
                f'{result.precision:.3f}',
                f'{result.recall:.3f}',
                f'{result.f1_score:.3f}'
            ])
        
        table = ax_table.table(cellText=table_data,
                              colLabels=['Model', 'Acc', 'Prec', 'Rec', 'F1'],
                              loc='center',
                              cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        for i in range(len(table_data) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#f0f0f0')
                    cell.set_text_props(weight='bold')
        
        ax_table.set_title('Summary Table', fontweight='bold', pad=20)
        
        # 6. Best Model Highlight (bottom right)
        ax_best = fig.add_subplot(gs[2, 2])
        ax_best.axis('off')
        
        best_model = max(results.keys(), 
                        key=lambda k: results[k].f1_score)
        best_result = results[best_model]
        
        summary_text = f"""
        BEST MODEL
        
        {best_model}
        
        F1-Score: {best_result.f1_score:.4f}
        Accuracy: {best_result.accuracy:.4f}
        Precision: {best_result.precision:.4f}
        Recall: {best_result.recall:.4f}
        
        Training: {best_result.training_time:.2f}s
        Prediction: {best_result.prediction_time*1000:.2f}ms
        """
        
        ax_best.text(0.5, 0.5, summary_text, 
                    fontsize=11, verticalalignment='center',
                    horizontalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', 
                             alpha=0.3, pad=1))
        
        fig.suptitle('Model Performance Dashboard - Sentiment Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Model dashboard saved to {save_path}")
        
        return fig


# Testing and demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("MODEL VISUALIZER MODULE")
    logger.info("\nAvailable visualization functions:")
    logger.info("  - plot_confusion_matrices()")
    logger.info("  - plot_roc_curves()")
    logger.info("  - plot_metrics_comparison()")
    logger.info("  - plot_training_times()")
    logger.info("  - create_comparison_table_styled()")
    logger.info("  - create_model_dashboard()")
    logger.info("\nUsage example:")
    logger.info("  from visualizations.model_viz import ModelVisualizer")
    logger.info("  viz = ModelVisualizer()")
    logger.info("  viz.create_model_dashboard(results, models, X_test, y_test)")
    logger.info("  viz.plot_roc_curves(trained_models, X_test, y_test)")