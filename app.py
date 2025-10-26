"""
Sentiment Analysis System - Streamlit Web Application
Provides interface for training models and making predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Import our modules
from src.data_loader import DataLoader
from src.text_preprocessor import TextPreprocessor
from src.models.sentiment_classifier import SentimentAnalyzer
from src.visualizations.data_viz import DataVisualizer
from src.visualizations.model_viz import ModelVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = TextPreprocessor(use_lemmatization=False)
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Train Models", "Make Predictions", "Model Comparison", "About"]
)

# Create directories if they don't exist
Path("artifacts").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)


# ==================== HOME PAGE ====================
if page == "Home":
    st.markdown('<p class="main-header">Sentiment Analysis System</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Amazon Product Review Sentiment Analysis System

    This application uses **Machine Learning** to analyze sentiment in product reviews using three different algorithms:
    - **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
    - **Logistic Regression** - Linear model for binary classification
    - **Random Forest** - Ensemble method using multiple decision trees

    ---

    #### How to Use This Application

    **1. Train Models**
    - Upload your CSV dataset with 'Text' and 'Score' columns
    - Configure training parameters (sample size, balance classes)
    - Train all three models and compare performance
    - Select the best model for deployment

    **2. Make Predictions**
    - Enter text directly or upload a CSV file
    - Get sentiment predictions (Positive/Negative) with confidence scores
    - Download results

    **3. Model Comparison**
    - View detailed performance metrics
    - Compare accuracy, precision, recall, F1-score, and ROC-AUC
    - Analyze confusion matrices and ROC curves

    ---

    #### Dataset Requirements
    Your CSV file should have:
    - **Text column**: Review text content
    - **Score column**: Star ratings (1-5)
        - 1-2 stars = Negative sentiment
        - 3 stars = Neutral (skipped)
        - 4-5 stars = Positive sentiment

    """)

    # Status indicators
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.trained:
            st.success("Models Trained")
        else:
            st.info("Models Not Trained")

    with col2:
        if st.session_state.selected_model:
            st.success(f"Deployed: {st.session_state.selected_model}")
        else:
            st.info("No Model Deployed")

    with col3:
        artifact_files = list(Path("artifacts").glob("*.pkl"))
        st.info(f"{len(artifact_files)} Saved Models")


# ==================== TRAIN MODELS PAGE ====================
elif page == "Train Models":
    st.markdown('<p class="main-header">Train Sentiment Analysis Models</p>', unsafe_allow_html=True)

    st.markdown("""
    Upload your dataset and train three machine learning models. The system will automatically:
    - Clean and preprocess the text data
    - Balance positive/negative classes
    - Train Naive Bayes, Logistic Regression, and Random Forest
    - Generate performance comparison visualizations
    """)

    # File upload
    st.markdown("---")
    st.subheader("Step 1: Upload Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with 'Text' and 'Score' columns"
    )

    # Training configuration
    st.markdown("---")
    st.subheader("Step 2: Configure Training Parameters")

    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.number_input(
            "Sample Size",
            min_value=1000,
            max_value=100000,
            value=20000,
            step=1000,
            help="Number of reviews to use for training (larger = slower but more accurate)"
        )

    with col2:
        balance_classes = st.checkbox(
            "Balance Classes",
            value=True,
            help="Ensure equal number of positive and negative samples"
        )

    vectorizer_type = st.selectbox(
        "Vectorization Method",
        ['tfidf', 'count'],
        help="TF-IDF: Weights words by importance | Count: Simple word frequency"
    )

    max_features = st.slider(
        "Maximum Features",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=1000,
        help="Maximum number of words to consider"
    )

    # Train button
    st.markdown("---")
    st.subheader("Step 3: Train Models")

    if uploaded_file is not None:
        if st.button("Train All Models", type="primary", use_container_width=True):
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Load data
                status_text.text("Loading data...")
                progress_bar.progress(10)

                # Save uploaded file temporarily
                temp_file = "data/temp_upload.csv"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Single flexible DataLoader works across all domains
                loader = DataLoader(temp_file)
                reviews, stats = loader.load_data(sample_size=sample_size, balance=balance_classes)

                # Display field extraction info
                field_info = stats['field_extraction']
                st.success(f"Loaded {len(reviews)} reviews")
                st.info(f"Dataset type: **{field_info['dataset_type']}** | "
                       f"Text field: **{field_info['text_column']}** | "
                       f"Sentiment field: **{field_info['sentiment_column']}**")

                # Step 2: Preprocess
                status_text.text("Preprocessing text...")
                progress_bar.progress(25)

                preprocessor = st.session_state.preprocessor
                texts = [preprocessor.preprocess(r.text) for r in reviews]
                labels = [r.sentiment for r in reviews]

                # Step 3: Create visualizations
                status_text.text("Creating data visualizations...")
                progress_bar.progress(35)

                df = pd.DataFrame({'Text': texts, 'sentiment': labels})
                data_viz = DataVisualizer()

                # Generate plots
                with st.spinner("Generating sentiment distribution..."):
                    data_viz.plot_sentiment_distribution(df, save_path='plots/sentiment_dist.png')
                with st.spinner("Generating review length analysis..."):
                    data_viz.plot_review_length_distribution(df, text_column='Text',
                                                            save_path='plots/length_dist.png')

                # Step 4: Train models
                status_text.text("Training machine learning models...")
                progress_bar.progress(50)

                analyzer = SentimentAnalyzer(vectorizer_type=vectorizer_type, max_features=max_features)

                # Split data
                texts_train, texts_test, labels_train, labels_test = train_test_split(
                    texts, labels, test_size=0.2, random_state=42, stratify=labels
                )

                # Prepare data
                status_text.text("Vectorizing text features...")
                progress_bar.progress(60)
                X_train, y_train = analyzer.prepare_data(texts_train, labels_train, fit_vectorizer=True)
                X_test, y_test = analyzer.prepare_data(texts_test, labels_test, fit_vectorizer=False)

                # Train all models
                status_text.text("Training Naive Bayes, Logistic Regression, and Random Forest...")
                progress_bar.progress(70)

                results = analyzer.train_all_models(X_train, y_train, X_test, y_test)

                # Step 5: Create model visualizations
                status_text.text("Creating model comparison visualizations...")
                progress_bar.progress(85)

                model_viz = ModelVisualizer()
                model_viz.plot_confusion_matrices(results, save_path='plots/confusion_matrices.png')
                model_viz.plot_roc_curves(analyzer.trained_models, X_test, y_test,
                                         save_path='plots/roc_curves.png')
                model_viz.plot_metrics_comparison(results, save_path='plots/metrics_comparison.png')
                model_viz.create_model_dashboard(results, analyzer.trained_models, X_test, y_test,
                                                save_path='plots/model_dashboard.png')

                # Step 6: Save models
                status_text.text("Saving trained models...")
                progress_bar.progress(95)

                for model_name in results.keys():
                    safe_name = model_name.replace(" ", "_").lower()
                    analyzer.save_model(model_name, f'artifacts/{safe_name}_model.pkl')

                # Complete
                progress_bar.progress(100)
                status_text.text("Training complete!")

                # Store in session state
                st.session_state.analyzer = analyzer
                st.session_state.trained = True
                st.session_state.results = results

                # Display results
                st.balloons()
                st.success("All models trained successfully!")

                # Show comparison table
                st.markdown("---")
                st.subheader("Model Performance Comparison")
                comparison_df = analyzer.get_comparison_dataframe()
                st.dataframe(comparison_df, use_container_width=True)

                # Show visualizations
                st.markdown("---")
                st.subheader("Performance Visualizations")

                col1, col2 = st.columns(2)
                with col1:
                    st.image('plots/confusion_matrices.png', caption='Confusion Matrices')
                with col2:
                    st.image('plots/metrics_comparison.png', caption='Metrics Comparison')

                st.image('plots/roc_curves.png', caption='ROC Curves', use_column_width=True)

                # Model selection for deployment
                st.markdown("---")
                st.subheader("Step 4: Select Model for Deployment")
                st.markdown("Choose which model to use for making predictions:")

                best_model_name, _ = analyzer.get_best_model(metric='f1_score')

                selected = st.selectbox(
                    "Select Model",
                    list(results.keys()),
                    index=list(results.keys()).index(best_model_name)
                )

                if st.button("Deploy Selected Model", type="primary"):
                    st.session_state.selected_model = selected
                    st.success(f"Deployed: {selected}")
                    st.info("Go to 'Make Predictions' page to use the deployed model!")

            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                logger.error(f"Training error: {e}", exc_info=True)
    else:
        st.info("Please upload a CSV file to begin training")


# ==================== MAKE PREDICTIONS PAGE ====================
elif page == "Make Predictions":
    st.markdown('<p class="main-header">Make Sentiment Predictions</p>', unsafe_allow_html=True)

    # Check if model is trained
    if not st.session_state.trained or st.session_state.analyzer is None:
        st.warning("No trained models available. Please go to 'Train Models' page first.")

        # Check for saved models
        st.markdown("---")
        st.subheader("Or Load a Saved Model")

        artifact_files = list(Path("artifacts").glob("*.pkl"))
        if artifact_files:
            selected_file = st.selectbox(
                "Select a saved model",
                [f.name for f in artifact_files]
            )

            if st.button("Load Model"):
                try:
                    analyzer = SentimentAnalyzer()
                    analyzer.load_model(f"artifacts/{selected_file}")
                    st.session_state.analyzer = analyzer
                    st.session_state.trained = True
                    st.session_state.selected_model = selected_file.replace("_model.pkl", "").replace("_", " ").title()
                    st.success(f"Loaded model: {selected_file}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        else:
            st.info("No saved models found in artifacts/ directory")

    else:
        # Model selection
        if st.session_state.selected_model:
            st.success(f"Using model: **{st.session_state.selected_model}**")
        else:
            st.info("Select a model to use for predictions:")
            model_choice = st.selectbox(
                "Select Model",
                list(st.session_state.results.keys()) if st.session_state.results else []
            )
            if st.button("Use This Model"):
                st.session_state.selected_model = model_choice
                st.rerun()

        st.markdown("---")

        # Prediction mode selection
        prediction_mode = st.radio(
            "Prediction Mode",
            ["Single Text", "Batch Upload (CSV)"],
            horizontal=True
        )

        st.markdown("---")

        # Single text prediction
        if prediction_mode == "Single Text":
            st.subheader("Enter Review Text")

            user_text = st.text_area(
                "Type or paste a product review:",
                height=150,
                placeholder="Example: This product is amazing! Best purchase ever. Highly recommend!"
            )

            if st.button("Predict Sentiment", type="primary", use_container_width=True):
                if user_text.strip():
                    try:
                        # Preprocess
                        preprocessed = st.session_state.preprocessor.preprocess(user_text)

                        # Predict
                        predictions = st.session_state.analyzer.predict(
                            [preprocessed],
                            model_name=st.session_state.selected_model
                        )

                        pred = predictions[0]

                        # Display result
                        st.markdown("---")
                        st.subheader("Prediction Result")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            sentiment_emoji = "positive" if pred['sentiment'] == 'positive' else "negative"
                            st.metric("Sentiment", f"{sentiment_emoji.upper()}")

                        with col2:
                            st.metric("Confidence", f"{pred['confidence']:.1%}")

                        with col3:
                            st.metric("Model Used", pred['model'])

                        # Visual confidence bar
                        st.markdown("**Confidence Breakdown:**")
                        if pred['sentiment'] == 'positive':
                            st.progress(pred['confidence'])
                            st.caption(f"Positive: {pred['confidence']:.1%} | Negative: {1-pred['confidence']:.1%}")
                        else:
                            st.progress(pred['confidence'])
                            st.caption(f"Negative: {pred['confidence']:.1%} | Positive: {1-pred['confidence']:.1%}")

                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                else:
                    st.warning("Please enter some text to analyze")

        # Batch CSV prediction
        else:
            st.subheader("Upload CSV for Batch Predictions")

            st.markdown("""
            Upload a CSV file with a **'Text'** column containing reviews to analyze.
            The system will predict sentiment for each review.
            """)

            batch_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                key='batch_upload'
            )

            if batch_file is not None:
                try:
                    # Load CSV
                    df_batch = pd.read_csv(batch_file)

                    st.success(f"Loaded {len(df_batch)} reviews")

                    # Verify column
                    if 'Text' not in df_batch.columns:
                        st.error("CSV must have a 'Text' column")
                    else:
                        # Show preview
                        st.markdown("**Preview:**")
                        st.dataframe(df_batch.head(), use_container_width=True)

                        if st.button("Predict All", type="primary", use_container_width=True):
                            with st.spinner("Making predictions..."):
                                # Preprocess all texts
                                texts = df_batch['Text'].astype(str).tolist()
                                preprocessed = [st.session_state.preprocessor.preprocess(t) for t in texts]

                                # Predict
                                predictions = st.session_state.analyzer.predict(
                                    preprocessed,
                                    model_name=st.session_state.selected_model
                                )

                                # Add predictions to dataframe
                                df_batch['Predicted_Sentiment'] = [p['sentiment'] for p in predictions]
                                df_batch['Confidence'] = [p['confidence'] for p in predictions]
                                df_batch['Model'] = [p['model'] for p in predictions]

                                st.success("Predictions complete!")

                                # Show results
                                st.markdown("---")
                                st.subheader("Prediction Results")
                                st.dataframe(df_batch, use_container_width=True)

                                # Summary statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    positive_count = (df_batch['Predicted_Sentiment'] == 'positive').sum()
                                    st.metric("Positive Reviews", positive_count)

                                with col2:
                                    negative_count = (df_batch['Predicted_Sentiment'] == 'negative').sum()
                                    st.metric("Negative Reviews", negative_count)

                                with col3:
                                    avg_conf = df_batch['Confidence'].mean()
                                    st.metric("Avg Confidence", f"{avg_conf:.1%}")

                                # Download button
                                csv = df_batch.to_csv(index=False)
                                st.download_button(
                                    label="Download Results CSV",
                                    data=csv,
                                    file_name="sentiment_predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )

                except Exception as e:
                    st.error(f"Error processing batch predictions: {str(e)}")


# ==================== MODEL COMPARISON PAGE ====================
elif page == "Model Comparison":
    st.markdown('<p class="main-header">Model Performance Comparison</p>', unsafe_allow_html=True)

    if not st.session_state.trained or st.session_state.results is None:
        st.warning("No training results available. Please train models first.")
    else:
        results = st.session_state.results

        # Comparison table
        st.subheader("Performance Metrics")
        comparison_df = st.session_state.analyzer.get_comparison_dataframe()
        st.dataframe(comparison_df, use_container_width=True)

        # Metric cards
        st.markdown("---")
        st.subheader("Detailed Model Metrics")

        for model_name, result in results.items():
            with st.expander(f"{model_name}", expanded=False):
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Accuracy", f"{result.accuracy:.4f}")
                with col2:
                    st.metric("Precision", f"{result.precision:.4f}")
                with col3:
                    st.metric("Recall", f"{result.recall:.4f}")
                with col4:
                    st.metric("F1-Score", f"{result.f1_score:.4f}")
                with col5:
                    if result.roc_auc:
                        st.metric("ROC-AUC", f"{result.roc_auc:.4f}")
                    else:
                        st.metric("ROC-AUC", "N/A")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Training Time", f"{result.training_time:.2f}s")
                with col_b:
                    st.metric("Prediction Time", f"{result.prediction_time*1000:.2f}ms")

        # Visualizations
        st.markdown("---")
        st.subheader("Performance Visualizations")

        # Check if plots exist
        if Path('plots/confusion_matrices.png').exists():
            st.image('plots/confusion_matrices.png', caption='Confusion Matrices', use_column_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if Path('plots/metrics_comparison.png').exists():
                st.image('plots/metrics_comparison.png', caption='Metrics Comparison')
        with col2:
            if Path('plots/roc_curves.png').exists():
                st.image('plots/roc_curves.png', caption='ROC Curves')

        if Path('plots/model_dashboard.png').exists():
            st.markdown("---")
            st.subheader("Comprehensive Dashboard")
            st.image('plots/model_dashboard.png', caption='Model Dashboard', use_column_width=True)

        # Best model recommendation
        st.markdown("---")
        st.subheader("Recommendation")
        best_model, _ = st.session_state.analyzer.get_best_model(metric='f1_score')
        best_result = results[best_model]

        st.info(f"""
        **Recommended Model: {best_model}**

        Based on F1-Score ({best_result.f1_score:.4f}), this model provides the best balance
        between precision and recall for sentiment classification.

        - Accuracy: {best_result.accuracy:.4f}
        - Training Time: {best_result.training_time:.2f}s
        - Prediction Speed: {best_result.prediction_time*1000:.2f}ms per batch
        """)


# ==================== ABOUT PAGE ====================
elif page == "About":
    st.markdown('<p class="main-header">About This Project</p>', unsafe_allow_html=True)

    st.markdown("""
    ## INSY 4325 - Business Analytics Project

    ### Project Overview
    This sentiment analysis system is a comprehensive machine learning application that analyzes
    Amazon product reviews to determine whether customer sentiment is positive or negative.

    ---

    ### System Information
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **Directories:**
        - Artifacts: {len(list(Path('artifacts').glob('*.pkl')))} saved models
        - Plots: {'exists' if Path('plots').exists() else 'not created'}
        - Data: {'exists' if Path('data').exists() else 'not found'}
        """)

    with col2:
        st.info(f"""
        **Session Status:**
        - Models Trained: {'Yes' if st.session_state.trained else 'No'}
        - Deployed Model: {st.session_state.selected_model or 'None'}
        - Preprocessor: {'Ready' if st.session_state.preprocessor else 'Not ready'}
        """)


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>"
    "Sentiment Analysis System | INSY 4325 Business Analytics Project"
    "</div>",
    unsafe_allow_html=True
)
