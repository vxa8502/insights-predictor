"""
Sentiment Analysis System with Three Machine Learning Algorithms
Implements Naive Bayes, Logistic Regression, and Random Forest classifiers
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score)  # Added roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class ModelResult:
    """Container for storing model evaluation results"""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    training_time: float
    prediction_time: float
    predictions: np.ndarray
    roc_auc: Optional[float] = None  # Added ROC-AUC field


class SentimentAnalyzer:
    """
    Multi-algorithm sentiment analysis system.
    Trains and compares three models: Naive Bayes, Logistic Regression, Random Forest.
    """
    
    def __init__(self, vectorizer_type: str = 'tfidf', max_features: int = 5000):
        """
        Initialize the sentiment analyzer.
        
        Args:
            vectorizer_type: 'tfidf' or 'count' for text vectorization
            max_features: Maximum number of features to extract from text
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize text vectorizer
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # Use unigrams and bigrams
                min_df=2,            # Ignore terms that appear in less than 2 documents
                max_df=0.95          # Ignore terms that appear in more than 95% of documents
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        
        # Initialize three classification models
        self.models = {
            'Naive Bayes': MultinomialNB(alpha=1.0),
            
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver='lbfgs',
                random_state=42
            ),
            
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=50,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
        }
        
        # Storage for trained models and results
        self.trained_models = {}
        self.results = {}
        self.label_encoder = {'negative': 0, 'positive': 1}
        self.label_decoder = {0: 'negative', 1: 'positive'}
        self.is_fitted = False
    
    def prepare_data(self, texts: List[str], labels: List[str], 
                    fit_vectorizer: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform text data into numerical features and encode labels.
        
        Args:
            texts: List of preprocessed text strings
            labels: List of sentiment labels ('positive' or 'negative')
            fit_vectorizer: If True, fit the vectorizer (use for training data only)
            
        Returns:
            Tuple of (vectorized features matrix, encoded labels array)
        """
        self.logger.info(f"Vectorizing {len(texts)} texts...")
        
        # Vectorize text
        if fit_vectorizer:
            X_vec = self.vectorizer.fit_transform(texts)
            self.logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Vectorizer not fitted. Call with fit_vectorizer=True first.")
            X_vec = self.vectorizer.transform(texts)
        
        # Encode labels to numeric values
        y_encoded = np.array([self.label_encoder[label] for label in labels])
        
        return X_vec, y_encoded
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, ModelResult]:
        """
        Train all three models and evaluate on test set.
        
        Args:
            X_train: Training features (vectorized)
            y_train: Training labels (encoded as 0/1)
            X_test: Test features (vectorized)
            y_test: Test labels (encoded as 0/1)
            
        Returns:
            Dictionary mapping model names to ModelResult objects
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("TRAINING ALL MODELS")
        self.logger.info("="*60)
        
        self.results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"\nTraining {name}...")
            
            # Train the model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions on test set
            start_time = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate ROC-AUC if model supports probability predictions
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]  # Get probability of positive class
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                # Fallback for models without predict_proba (shouldn't happen with our models)
                roc_auc = None
                self.logger.warning(f"{name} doesn't support probability predictions for ROC-AUC")
            
            # Store results
            result = ModelResult(
                name=name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=cm,
                training_time=training_time,
                prediction_time=prediction_time,
                predictions=y_pred,
                roc_auc=roc_auc  # Store ROC-AUC
            )
            
            self.results[name] = result
            self.trained_models[name] = model
            
            # Log performance metrics
            self.logger.info(f"\n{name} Performance:")
            self.logger.info(f"  Accuracy:  {accuracy:.4f}")
            self.logger.info(f"  Precision: {precision:.4f}")
            self.logger.info(f"  Recall:    {recall:.4f}")
            self.logger.info(f"  F1-Score:  {f1:.4f}")
            if roc_auc is not None:
                self.logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
            self.logger.info(f"  Training time: {training_time:.2f}s")
            self.logger.info(f"  Prediction time: {prediction_time:.4f}s")
        
        return self.results
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Any]:
        """
        Identify the best performing model based on specified metric.
        
        Args:
            metric: Metric to use ('accuracy', 'precision', 'recall', 'f1_score', 'roc_auc')
            
        Returns:
            Tuple of (best model name, best model object)
        """
        if not self.results:
            raise ValueError("No models trained yet. Call train_all_models() first.")
        
        # Filter out models with None values for the specified metric (e.g., ROC-AUC)
        valid_results = {k: v for k, v in self.results.items() 
                        if getattr(v, metric) is not None}
        
        if not valid_results:
            raise ValueError(f"No models have valid {metric} scores")
        
        best_name = max(valid_results.keys(), 
                       key=lambda k: getattr(valid_results[k], metric))
        best_score = getattr(self.results[best_name], metric)
        
        self.logger.info(f"\nBest model by {metric}: {best_name} ({best_score:.4f})")
        
        return best_name, self.trained_models[best_name]
    
    def predict(self, texts: List[str], model_name: str = None) -> List[Dict]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of preprocessed text strings
            model_name: Name of model to use (if None, uses best F1 model)
            
        Returns:
            List of dictionaries with 'sentiment', 'confidence', and 'model' keys
        """
        if not self.trained_models:
            raise ValueError("No models trained. Call train_all_models() first.")
        
        # Select model
        if model_name is None:
            model_name, model = self.get_best_model(metric='f1_score')
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model '{model_name}' not found. Available: {list(self.trained_models.keys())}")
            model = self.trained_models[model_name]
        
        # Vectorize input texts
        X_vec = self.vectorizer.transform(texts)
        
        # Get predictions
        predictions = model.predict(X_vec)
        
        # Get confidence scores (probability of predicted class)
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_vec)
            confidences = np.max(probas, axis=1)
        else:
            # Random Forest has predict_proba, but fallback just in case
            confidences = np.ones(len(predictions))
        
        # Format results
        results = []
        for pred, conf in zip(predictions, confidences):
            results.append({
                'sentiment': self.label_decoder[pred],
                'confidence': float(conf),
                'model': model_name
            })
        
        return results
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Create a comparison table of all model performances.
        
        Returns:
            Pandas DataFrame with model comparison metrics
        """
        if not self.results:
            raise ValueError("No models trained yet.")
        
        data = []
        for name, result in self.results.items():
            row_data = {
                'Model': name,
                'Accuracy': f"{result.accuracy:.4f}",
                'Precision': f"{result.precision:.4f}",
                'Recall': f"{result.recall:.4f}",
                'F1-Score': f"{result.f1_score:.4f}",
                'Training Time (s)': f"{result.training_time:.2f}",
                'Prediction Time (s)': f"{result.prediction_time:.4f}"
            }
            
            # Add ROC-AUC if available
            if result.roc_auc is not None:
                row_data['ROC-AUC'] = f"{result.roc_auc:.4f}"
            else:
                row_data['ROC-AUC'] = "N/A"
            
            data.append(row_data)
        
        df = pd.DataFrame(data)
        # Sort by F1-Score descending
        df['_sort_key'] = [self.results[row['Model']].f1_score for _, row in df.iterrows()]
        df = df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)
        df = df.reset_index(drop=True)
        
        return df
    
    def print_detailed_report(self, y_test: np.ndarray):
        """
        Print detailed classification reports for all models.
        
        Args:
            y_test: True test labels (encoded)
        """
        print("\n")
        print("DETAILED CLASSIFICATION REPORTS")
        
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(classification_report(y_test, result.predictions, 
                                       target_names=['negative', 'positive']))
            if result.roc_auc is not None:
                print(f"ROC-AUC Score: {result.roc_auc:.4f}")
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk with vectorizer.
        
        Args:
            model_name: Name of the model to save
            filepath: Path where to save the model (e.g., 'artifacts/best_model.pkl')
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_data = {
            'model': self.trained_models[model_name],
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'model_name': model_name,
            'metrics': {
                'accuracy': self.results[model_name].accuracy,
                'precision': self.results[model_name].precision,
                'recall': self.results[model_name].recall,
                'f1_score': self.results[model_name].f1_score,
                'roc_auc': self.results[model_name].roc_auc
            }
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model '{model_name}' saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file
        """
        model_data = joblib.load(filepath)
        
        model_name = model_data['model_name']
        self.trained_models[model_name] = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.label_decoder = model_data['label_decoder']
        self.is_fitted = True
        
        self.logger.info(f"Model '{model_name}' loaded from {filepath}")
        if 'metrics' in model_data:
            self.logger.info(f"Model metrics: {model_data['metrics']}")


# Standalone testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("SENTIMENT ANALYZER - READY")
    logger.info("\nThis module provides a complete sentiment analysis system with:")
    logger.info("  - Three ML algorithms (Naive Bayes, Logistic Regression, Random Forest)")
    logger.info("  - Automatic model comparison and selection")
    logger.info("  - Prediction with confidence scores")
    logger.info("  - ROC-AUC evaluation metric")
    logger.info("  - Model persistence (save/load)")
    logger.info("\nIntegrate with your data_loader.py and text_preprocessor.py:")
    logger.info("  from sentiment_classifier import SentimentAnalyzer")
    logger.info("  analyzer = SentimentAnalyzer(vectorizer_type='tfidf')")
    logger.info("  X_train, y_train = analyzer.prepare_data(train_texts, train_labels)")
    logger.info("  X_test, y_test = analyzer.prepare_data(test_texts, test_labels, fit_vectorizer=False)")
    logger.info("  results = analyzer.train_all_models(X_train, y_train, X_test, y_test)")
    logger.info("  best_model_name, best_model = analyzer.get_best_model(metric='roc_auc')")