"""
Cross-Domain Sentiment Analysis
================================
Trains models on each domain and tests across ALL domains to find:
1. Which domain creates the best universal model
2. How well models generalize across domains
3. Domain-specific vocabulary and patterns
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import DataLoader
from src.text_preprocessor import TextPreprocessor
from src.models.sentiment_classifier import SentimentAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrossDomainAnalyzer:
    """
    Systematically trains on each domain and tests across all domains.
    Creates comprehensive cross-domain performance matrix.
    """

    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.preprocessor = TextPreprocessor(use_lemmatization=False)
        self.domain_data = {}  # Store processed data for each domain
        self.models = {}       # Store trained models for each domain
        self.results_matrix = {}  # Performance matrix: train_domain -> test_domain -> metrics

        # Create output directories
        Path("cross_domain_plots").mkdir(exist_ok=True)

    def find_datasets(self):
        """Find all CSV and TSV files in datasets directory."""
        if not self.datasets_dir.exists():
            logger.error(f"Datasets directory '{self.datasets_dir}' not found!")
            logger.info("Please create 'datasets/' folder and add your CSV/TSV files")
            return []

        csv_datasets = list(self.datasets_dir.glob("*.csv"))
        tsv_datasets = list(self.datasets_dir.glob("*.tsv"))
        datasets = csv_datasets + tsv_datasets

        logger.info(f"Found {len(datasets)} datasets:")
        for d in datasets:
            size_mb = d.stat().st_size / (1024 * 1024)
            logger.info(f"  - {d.name} ({size_mb:.1f} MB)")
        return datasets

    def load_domain_data(self, dataset_path: Path, sample_size: int = 5000):
        """
        Load and preprocess data for a single domain.

        Args:
            dataset_path: Path to CSV file
            sample_size: Max samples to use per domain

        Returns:
            Dictionary with domain info and processed data
        """
        domain_name = dataset_path.stem.replace('_', ' ').title()
        logger.info(f"\n{'='*80}")
        logger.info(f"Loading domain: {domain_name}")
        logger.info(f"{'='*80}")

        try:
            # Load data
            loader = DataLoader(str(dataset_path))
            reviews, stats = loader.load_data(sample_size=sample_size, balance=True)

            logger.info(f"✓ Loaded {len(reviews)} balanced reviews")
            logger.info(f"  Text column: '{stats['field_extraction']['text_column']}'")
            logger.info(f"  Sentiment column: '{stats['field_extraction']['sentiment_column']}'")

            # Preprocess
            texts = [self.preprocessor.preprocess(r.text) for r in reviews]
            labels = [r.sentiment for r in reviews]

            # Train/test split (80/20)
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )

            domain_info = {
                'name': domain_name,
                'path': str(dataset_path),
                'total_samples': len(reviews),
                'train_samples': len(X_train_text),
                'test_samples': len(X_test_text),
                'X_train_text': X_train_text,
                'X_test_text': X_test_text,
                'y_train': y_train,
                'y_test': y_test,
                'stats': stats
            }

            logger.info(f"✓ Split: {len(X_train_text)} train, {len(X_test_text)} test")
            return domain_info

        except Exception as e:
            logger.error(f"✗ Failed to load {domain_name}: {e}")
            return None

    def train_domain_model(self, domain_name: str, domain_data: dict):
        """
        Train a model on a specific domain.

        Args:
            domain_name: Name of the domain
            domain_data: Processed domain data

        Returns:
            Trained SentimentAnalyzer
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Training model on: {domain_name}")
        logger.info(f"{'='*80}")

        # Initialize analyzer
        analyzer = SentimentAnalyzer(vectorizer_type='tfidf', max_features=5000)

        # Prepare data
        X_train, y_train = analyzer.prepare_data(
            domain_data['X_train_text'],
            domain_data['y_train'],
            fit_vectorizer=True
        )

        # Train all three models
        X_test, y_test = analyzer.prepare_data(
            domain_data['X_test_text'],
            domain_data['y_test'],
            fit_vectorizer=False
        )

        results = analyzer.train_all_models(X_train, y_train, X_test, y_test)

        # Get best model
        best_name, best_model = analyzer.get_best_model(metric='f1_score')
        best_f1 = results[best_name].f1_score  # Access attribute, not dict key
        logger.info(f"✓ Best model: {best_name} (F1: {best_f1:.3f})")

        # Store best model name for later use
        analyzer.best_model_name = best_name

        return analyzer

    def test_cross_domain(self, train_domain: str, test_domain: str,
                         model_analyzer: SentimentAnalyzer, test_data: dict):
        """
        Test a model trained on one domain against another domain.

        Args:
            train_domain: Domain model was trained on
            test_domain: Domain to test on
            model_analyzer: Trained analyzer
            test_data: Test domain data

        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"  Testing {train_domain} model on {test_domain}...")

        # Prepare test data using the trained vectorizer
        X_test, y_test = model_analyzer.prepare_data(
            test_data['X_test_text'],
            test_data['y_test'],
            fit_vectorizer=False
        )

        # Get predictions from best model
        best_model_name = model_analyzer.best_model_name
        model = model_analyzer.trained_models[best_model_name]

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Convert labels to binary if they're strings
        if isinstance(y_test[0], str):
            y_test_binary = [1 if y == 'positive' else 0 for y in y_test]
            y_pred_binary = [1 if y == 'positive' else 0 for y in y_pred]
        else:
            y_test_binary = y_test
            y_pred_binary = y_pred

        metrics = {
            'accuracy': accuracy_score(y_test_binary, y_pred_binary),
            'precision': precision_score(y_test_binary, y_pred_binary),
            'recall': recall_score(y_test_binary, y_pred_binary),
            'f1_score': f1_score(y_test_binary, y_pred_binary),
            'roc_auc': roc_auc_score(y_test_binary, y_pred_proba),
            'model_used': best_model_name,
            'test_samples': len(y_test)
        }

        logger.info(f"    → F1: {metrics['f1_score']:.3f}, Accuracy: {metrics['accuracy']:.3f}")
        return metrics

    def run_full_analysis(self, sample_size: int = 5000):
        """
        Run complete cross-domain analysis:
        1. Load all domains
        2. Train model on each domain
        3. Test each model on ALL domains
        4. Generate comprehensive report
        """
        datasets = self.find_datasets()
        if not datasets:
            return

        logger.info(f"\n{'#'*80}")
        logger.info(f"CROSS-DOMAIN SENTIMENT ANALYSIS")
        logger.info(f"Datasets: {len(datasets)} | Sample size per domain: {sample_size}")
        logger.info(f"{'#'*80}")

        # Step 1: Load all domains
        logger.info("\n[STEP 1] Loading all domains...")
        for dataset_path in datasets:
            domain_data = self.load_domain_data(dataset_path, sample_size)
            if domain_data:
                self.domain_data[domain_data['name']] = domain_data

        if not self.domain_data:
            logger.error("No domains loaded successfully. Exiting.")
            return

        domain_names = list(self.domain_data.keys())
        logger.info(f"\n✓ Successfully loaded {len(domain_names)} domains: {domain_names}")

        # Step 2: Train model on each domain
        logger.info("\n[STEP 2] Training models on each domain...")
        for domain_name in domain_names:
            model = self.train_domain_model(domain_name, self.domain_data[domain_name])
            self.models[domain_name] = model

        # Step 3: Test each model on ALL domains (including itself)
        logger.info("\n[STEP 3] Cross-domain testing (train on X, test on Y)...")
        for train_domain in domain_names:
            self.results_matrix[train_domain] = {}
            logger.info(f"\nModel trained on: {train_domain}")

            for test_domain in domain_names:
                metrics = self.test_cross_domain(
                    train_domain,
                    test_domain,
                    self.models[train_domain],
                    self.domain_data[test_domain]
                )
                self.results_matrix[train_domain][test_domain] = metrics

        # Step 4: Generate reports
        logger.info("\n[STEP 4] Generating analysis reports...")
        self.save_results()
        self.generate_visualizations()
        self.print_summary()

    def save_results(self):
        """Save results to JSON."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'domains': list(self.domain_data.keys()),
            'results_matrix': self.results_matrix,
            'domain_info': {
                name: {
                    'train_samples': data['train_samples'],
                    'test_samples': data['test_samples'],
                    'text_column': data['stats']['field_extraction']['text_column'],
                    'sentiment_column': data['stats']['field_extraction']['sentiment_column']
                }
                for name, data in self.domain_data.items()
            }
        }

        with open('cross_domain_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        logger.info("✓ Results saved to: cross_domain_results.json")

    def generate_visualizations(self):
        """Generate heatmaps and plots."""
        domain_names = list(self.domain_data.keys())

        # Create F1 score matrix
        f1_matrix = np.zeros((len(domain_names), len(domain_names)))
        for i, train_domain in enumerate(domain_names):
            for j, test_domain in enumerate(domain_names):
                f1_matrix[i][j] = self.results_matrix[train_domain][test_domain]['f1_score']

        # Heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=domain_names, yticklabels=domain_names,
                   vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': 'F1 Score'})
        ax.set_xlabel('Test Domain', fontweight='bold', fontsize=12)
        ax.set_ylabel('Train Domain', fontweight='bold', fontsize=12)
        ax.set_title('Cross-Domain Performance Matrix (F1 Score)\nTrain on Row, Test on Column',
                    fontweight='bold', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('cross_domain_plots/performance_heatmap.png', dpi=300, bbox_inches='tight')
        logger.info("✓ Saved: cross_domain_plots/performance_heatmap.png")

        # Best universal model bar chart
        avg_performance = {
            train_domain: np.mean([
                self.results_matrix[train_domain][test_domain]['f1_score']
                for test_domain in domain_names
            ])
            for train_domain in domain_names
        }

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(domain_names, [avg_performance[d] for d in domain_names],
                     color='#3498db', alpha=0.8, edgecolor='black')

        # Highlight best
        best_domain = max(avg_performance, key=avg_performance.get)
        best_idx = domain_names.index(best_domain)
        bars[best_idx].set_color('#2ecc71')

        ax.set_xlabel('Training Domain', fontweight='bold')
        ax.set_ylabel('Average F1 Score Across All Test Domains', fontweight='bold')
        ax.set_title('Which Domain Creates the Best Universal Model?', fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('cross_domain_plots/best_universal_model.png', dpi=300, bbox_inches='tight')
        logger.info("✓ Saved: cross_domain_plots/best_universal_model.png")

    def print_summary(self):
        """Print comprehensive summary."""
        domain_names = list(self.domain_data.keys())

        logger.info(f"\n{'='*100}")
        logger.info("CROSS-DOMAIN ANALYSIS SUMMARY")
        logger.info(f"{'='*100}")

        # Matrix table
        print(f"\nPerformance Matrix (F1 Scores):")
        print(f"{'Train \\ Test':<20}", end='')
        for test_domain in domain_names:
            print(f"{test_domain:<12}", end='')
        print()
        print("-" * (20 + 12 * len(domain_names)))

        for train_domain in domain_names:
            print(f"{train_domain:<20}", end='')
            for test_domain in domain_names:
                f1 = self.results_matrix[train_domain][test_domain]['f1_score']
                print(f"{f1:<12.3f}", end='')
            print()

        # Best universal model
        avg_performance = {
            train_domain: np.mean([
                self.results_matrix[train_domain][test_domain]['f1_score']
                for test_domain in domain_names
            ])
            for train_domain in domain_names
        }

        best_domain = max(avg_performance, key=avg_performance.get)
        worst_domain = min(avg_performance, key=avg_performance.get)

        print(f"\n{'='*100}")
        print("KEY FINDINGS:")
        print(f"{'='*100}")
        print(f"🏆 BEST UNIVERSAL MODEL: {best_domain}")
        print(f"   Average F1 across all domains: {avg_performance[best_domain]:.3f}")
        print(f"\n📉 Weakest generalizer: {worst_domain}")
        print(f"   Average F1 across all domains: {avg_performance[worst_domain]:.3f}")

        # In-domain vs cross-domain
        in_domain_avg = np.mean([
            self.results_matrix[d][d]['f1_score'] for d in domain_names
        ])
        cross_domain_scores = [
            self.results_matrix[train][test]['f1_score']
            for train in domain_names
            for test in domain_names
            if train != test
        ]
        cross_domain_avg = np.mean(cross_domain_scores)

        print(f"\n📊 IN-DOMAIN PERFORMANCE (train & test on same domain):")
        print(f"   Average F1: {in_domain_avg:.3f}")
        print(f"\n🔀 CROSS-DOMAIN PERFORMANCE (train on X, test on Y):")
        print(f"   Average F1: {cross_domain_avg:.3f}")
        print(f"   Generalization gap: {in_domain_avg - cross_domain_avg:.3f}")

        print(f"\n{'='*100}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-domain sentiment analysis")
    parser.add_argument('--datasets-dir', default='datasets', help='Directory with CSV datasets')
    parser.add_argument('--sample-size', type=int, default=5000, help='Samples per domain')
    args = parser.parse_args()

    analyzer = CrossDomainAnalyzer(datasets_dir=args.datasets_dir)
    analyzer.run_full_analysis(sample_size=args.sample_size)


if __name__ == "__main__":
    main()
