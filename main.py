"""
Main training script that integrates all components.
Run this to train models and generate comparison reports.
"""

import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data_loader import AmazonDataLoader, ReviewData
from src.text_preprocessor import TextPreprocessor
from src.models.sentiment_classifier import SentimentAnalyzer
from src.visualizations.data_viz import DataVisualizer
from src.visualizations.model_viz import ModelVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create output directories
    Path("plots").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    
    # 1. Load data
    logger.info("Step 1: Loading data...")
    loader = AmazonDataLoader('data/Reviews.csv')
    reviews, stats = loader.load_data(sample_size=20000, balance=True)
    
    # 2. Preprocess text
    logger.info("Step 2: Preprocessing text...")
    preprocessor = TextPreprocessor(use_lemmatization=False)
    texts = [preprocessor.preprocess(r.text) for r in reviews]
    labels = [r.sentiment for r in reviews]
    
    # 3. Create visualizations (data exploration)
    logger.info("Step 3: Creating data visualizations...")
    data_viz = DataVisualizer()
    # Convert to DataFrame for visualization
    import pandas as pd
    df = pd.DataFrame({'Text': texts, 'sentiment': labels})
    
    # Fixed: Properly pass named arguments
    data_viz.plot_sentiment_distribution(df, save_path='plots/sentiment_dist.png')
    data_viz.plot_review_length_distribution(df, text_column='Text', save_path='plots/length_dist.png')
    # Note: wordclouds work better with original text, so let's create another df with original text
    df_original = pd.DataFrame({'Text': [r.text for r in reviews], 'sentiment': labels})
    data_viz.generate_wordclouds(df_original, text_column='Text', save_path_prefix='plots/wordcloud')
    data_viz.plot_top_words(df, text_column='Text', save_path='plots/top_words.png')
    
    # 4. Train models
    logger.info("Step 4: Training models...")
    analyzer = SentimentAnalyzer(vectorizer_type='tfidf', max_features=5000)
    
    # Split data
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Prepare data
    X_train, y_train = analyzer.prepare_data(texts_train, labels_train, fit_vectorizer=True)
    X_test, y_test = analyzer.prepare_data(texts_test, labels_test, fit_vectorizer=False)
    
    # Train all models
    results = analyzer.train_all_models(X_train, y_train, X_test, y_test)
    
    # 5. Create model comparison visualizations
    logger.info("Step 5: Creating model comparison visualizations...")
    model_viz = ModelVisualizer()
    
    model_viz.plot_confusion_matrices(results, save_path='plots/confusion_matrices.png')
    model_viz.plot_roc_curves(analyzer.trained_models, X_test, y_test, save_path='plots/roc_curves.png')
    model_viz.plot_metrics_comparison(results, save_path='plots/metrics_comparison.png')
    model_viz.create_model_dashboard(results, analyzer.trained_models, X_test, y_test, 
                                     save_path='plots/model_dashboard.png')
    
    # 6. Print comparison table
    logger.info("\nModel Comparison:")
    comparison_df = analyzer.get_comparison_dataframe()
    print(comparison_df)
    
    # 7. Save best model
    best_name, _ = analyzer.get_best_model(metric='f1_score')
    analyzer.save_model(best_name, f'artifacts/{best_name.replace(" ", "_").lower()}_model.pkl')
    
    # 8. Print detailed classification report
    logger.info("\nDetailed Classification Reports:")
    analyzer.print_detailed_report(y_test)
    
    # 9. Test predictions on sample texts
    logger.info("\nTesting predictions on sample texts...")
    sample_texts = [
        "This product is amazing! Best purchase ever.",
        "Terrible quality. Complete waste of money.",
        "It's okay, nothing special but does the job."
    ]
    
    # Preprocess sample texts
    sample_preprocessed = [preprocessor.preprocess(text) for text in sample_texts]
    predictions = analyzer.predict(sample_preprocessed)
    
    logger.info("\nSample Predictions:")
    for text, pred in zip(sample_texts, predictions):
        logger.info(f"Text: '{text[:50]}...'")
        logger.info(f"  → Sentiment: {pred['sentiment']} (confidence: {pred['confidence']:.2f})")
    
    logger.info("\n✔ Training complete! Check 'plots/' and 'artifacts/' directories.")

if __name__ == "__main__":
    main()