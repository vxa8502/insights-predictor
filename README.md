# Sentiment Analysis System - INSY 4325 Project

**Business Analytics Project - Option 2: Predictive Analytics System**

Universal sentiment analysis system using Machine Learning. This system works with multiple review dataset formats, trains and compares three algorithms (Naive Bayes, Logistic Regression, Random Forest), and provides a web interface for deploying the best model to predict sentiment on new reviews.

## Key Features

- **Multi-Dataset Support:** Works with Amazon, hotel, airline, restaurant, theme park, and other review datasets
- **Automatic Field Detection:** Intelligently identifies text and rating columns across different naming conventions
- **Smart Validation:** Catches incompatible datasets early with clear, helpful error messages
- **Flexible Rating Scales:** Handles 1-5 stars, 1-10 ratings, 0-100 scores, and text labels
- **Three ML Algorithms:** Naive Bayes, Logistic Regression, Random Forest with full comparison
- **Web Interface:** Professional Streamlit UI for training, prediction, and model comparison

## Setup Instructions

### 1. Clone the Repository
Open your terminal app and type this:
```bash
git clone https://github.com/vxa8502/insights-predictor
cd insights-predictor
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
Test the training pipeline:
```bash
python main.py
```

This should train models and generate plots in the `plots/` folder. Open this folder on you laptop and verify the visuals. 

### 5. Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure
```
├── src/                  # Core ML code
├── data/                 # Dataset
├── artifacts/            # Trained models (generated)
├── plots/                # Visualizations (generated)
├── app.py               # Streamlit UI
├── main.py              # Training script
└── requirements.txt     # Dependencies
```

## Dataset

### Supported Dataset Formats

This system automatically works with multiple review dataset formats:

| Dataset Type | Text Column Examples | Rating Column Examples | Status |
|--------------|---------------------|------------------------|--------|
| **Amazon Reviews** | Text, review | Score, rating | Tested |
| **Hotel Reviews** | reviews.text, Review | reviews.rating, Rating | Tested |
| **Airline Reviews** | Review, review_text | Overall_Rating, rating | Tested |
| **Restaurant Reviews** | review, comment | rating, stars | Tested |
| **Theme Park Reviews** | Review_Text, text | Rating, score | Tested |

**Automatic Field Detection:**
- System tries 12+ common text field names: text, review, comment, content, etc.
- System tries 11+ common rating field names: rating, score, sentiment, overall, etc.
- Falls back to column position if names don't match

**Supported Rating Formats:**
- **1-5 star scale:** 1-2 = negative, 3 = neutral (skipped), 4-5 = positive
- **1-10 rating scale:** 1-4 = negative, 5-6 = neutral (skipped), 7-10 = positive
- **Star text format:** "1 star", "5 stars" (automatically parsed)
- **0-100 percentage:** 0-49 = negative, 50-69 = neutral, 70-100 = positive
- **Text labels:** "positive", "negative", "pos", "neg"
- **Binary:** 0/1, True/False

### Download Dataset
The dataset we have been using is NOT in the repo (because it's too large).

Download the Amazon Reviews dataset and place the CSV file in `data/Reviews.csv`.

[The link was shared on Teams.
](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset)

You can also use any other review dataset - the system will automatically detect the columns!

### Dataset Validation

The system validates datasets before training and provides clear error messages:
- Checks text field contains actual text (not numeric IDs)
- Verifies rating field uses a supported format
- Ensures minimum data quality and quantity
- Shows helpful suggestions if dataset is incompatible

## Usage

### Train Models
1. Go to "Train Models" page
2. Upload any review CSV dataset (system auto-detects columns)
3. Configure parameters (sample size, vectorization method)
4. Click "Train All Models"
5. View performance comparison and visualizations
6. Select best model for deployment

### Make Predictions
1. Go to "Make Predictions" page
2. Enter text or upload CSV
3. Click "Predict"
4. Download results


## Requirements
- Python 3.8+
- See `requirements.txt` for packages

---

## Assignment Component Mapping (INSY 4325 Option 2)

This project fulfills all 6 required components for the Business Analytics predictive system:

### [DONE] Component 1: Problem Domain Description
**Location:** README.md (below) + app.py "About" page

**Business Problem:**
E-commerce platforms receive thousands of product reviews daily. Manually analyzing customer sentiment is time-consuming and inconsistent. This automated sentiment analysis system helps businesses:
- Quickly identify dissatisfied customers requiring immediate attention
- Track product satisfaction trends over time
- Make data-driven decisions about product improvements
- Prioritize customer service responses

**Dataset Details:**
- **Primary Source:** Amazon Product Reviews dataset
- **Size:** ~500,000 reviews across multiple product categories
- **Also Compatible With:** Hotel reviews, airline reviews, restaurant reviews, theme park reviews
- **Features Used:**
  - Text field: Customer review content (natural language)
  - Rating field: Star ratings or numeric scores (auto-detected)
- **Target Variable:** Binary sentiment (Positive/Negative)
  - System automatically maps various rating scales to positive/negative
  - Neutral ratings excluded for clearer classification
- **Challenges:**
  - Highly imbalanced (more positive than negative) → Solved with class balancing
  - Variable text length (10-1000+ characters) → Handled by TF-IDF
  - Contains HTML, special characters, spelling errors → Cleaned by preprocessing
  - Sarcasm and context-dependent language → Partially addressed with bigrams
  - Multiple dataset formats → Solved with flexible field detection

### [DONE] Component 2: User Interface Layout/Design
**Location:** `app.py` (Streamlit web application)

**Interface Structure:**

**Page 1: Home Dashboard**
- Welcome message and system overview
- Status indicators (models trained, deployed model, saved artifacts)
- Quick navigation to key features
- Dataset requirements specification

**Page 2: Train Models**
- File upload widget (CSV dataset)
- Training configuration panel:
  - Sample size slider (1,000 - 100,000)
  - Class balancing toggle
  - Vectorization method selection (TF-IDF vs Count)
  - Max features slider (1,000 - 10,000)
- Progress bar with status updates during training
- Real-time training logs
- Performance comparison table
- Confusion matrices and ROC curves visualization
- Model selection dropdown for deployment

**Page 3: Make Predictions**
- Model status indicator (which model is deployed)
- Two prediction modes (radio button toggle):
  - **Single Text Mode:**
    - Text area for manual input
    - Predict button
    - Result display with sentiment, confidence percentage, visual progress bar
  - **Batch CSV Mode:**
    - File uploader
    - Data preview table
    - Predict button for batch processing
    - Results table with sentiment, confidence per review
    - Summary statistics (positive count, negative count, avg confidence)
    - Download CSV button for results

**Page 4: Model Comparison**
- Performance metrics table (all models)
- Expandable sections per model with detailed metrics
- Confusion matrix visualizations
- ROC curve comparisons
- Metrics bar charts
- Best model recommendation panel

**Page 5: About**
- Project documentation
- Technical methodology explanation
- Assignment component mapping
- System status information

**Design Principles:**
- Clean, professional layout with consistent color scheme
- Blue (#1f77b4) for headers and primary actions
- Green for positive sentiment, Red for negative
- Progress indicators for long-running operations
- Responsive columns for metric displays
- Download buttons for result export

### [DONE] Component 3: Data Cleaning Techniques
**Location:** `src/data_loader.py` + `src/text_preprocessor.py` + `src/field_extractor.py`

**Techniques Implemented:**

1. **Automatic Field Detection & Validation** (`field_extractor.py` + `data_loader.py`)
   - Flexible field mapping system tries 12+ text field names
   - Tries 11+ sentiment/rating field names
   - Validates text field contains actual text (not numeric IDs)
   - Validates rating field uses supported format
   - Provides clear error messages for incompatible datasets

2. **Multi-Scale Rating Normalization** (`data_loader.py:295-340`)
   - Handles 1-5 star ratings, 1-10 scales, 0-100 percentages
   - Parses text formats like "5 stars", "1 star"
   - Converts text labels (positive/negative)
   - Handles binary values (0/1, True/False)
   - Skips neutral ratings (3 stars, 5-6 on 10-scale, 50-69 on 100-scale)

3. **Null/Empty Text Handling** (`data_loader.py:125-128`)
   - Removes reviews with null text
   - Filters empty strings after whitespace removal
   - Tracks removal statistics

4. **Class Balancing** (`data_loader.py:215-239`)
   - Equalizes positive and negative sample counts
   - Prevents model bias toward majority class
   - Uses stratified sampling with random seed for reproducibility
   - Caps at 50k per class for performance

5. **Data Quality Validation** (`data_loader.py:157-293`)
   - Checks text field has meaningful content (avg length > 5 chars)
   - Validates sentiment field has >10% valid values
   - Ensures final dataset has minimum 100 reviews
   - Provides diagnostic statistics when validation fails

6. **HTML Tag Removal** (`text_preprocessor.py`)
   - Strips HTML markup using regex
   - Prevents tag text from influencing predictions

7. **Text Normalization** (`text_preprocessor.py`)
   - Lowercase conversion for consistency
   - Punctuation removal (except apostrophes for contractions)
   - Whitespace normalization

8. **Stop Word Removal** (`text_preprocessor.py`)
   - Filters 60+ common non-informative words
   - Preserves sentiment-bearing words like "not", "but"
   - Removes product-neutral terms like "product", "item"

9. **Short Word Filtering** (`text_preprocessor.py`)
   - Removes words with <3 characters
   - Reduces noise from abbreviations and typos

### [DONE] Component 4: Visualization Techniques
**Location:** `src/visualizations/data_viz.py` + `src/visualizations/model_viz.py`

**Data Exploration Visualizations:**

1. **Sentiment Distribution Bar Chart** (`data_viz.py:40-89`)
   - Shows class balance with counts and percentages
   - Color-coded bars (green=positive, red=negative)
   - Value labels on bars

2. **Review Length Analysis** (`data_viz.py:146-197`)
   - Box plots comparing length by sentiment
   - Overlapping histograms showing distribution differences
   - Identifies correlation between review length and sentiment

3. **Word Clouds** (`data_viz.py:91-144`)
   - Separate clouds for positive and negative reviews
   - Visualizes most frequent terms by sentiment
   - Uses color gradients (Greens for positive, Reds for negative)

4. **Top Words Bar Charts** (`data_viz.py:199-251`)
   - Horizontal bar charts of 20 most common words per sentiment
   - Frequency counts displayed
   - Side-by-side comparison

**Model Performance Visualizations:**

5. **Confusion Matrices** (`model_viz.py:38-87`)
   - Heatmaps for each model
   - Annotated with counts and percentages
   - Shows true vs predicted classifications

6. **ROC Curves** (`model_viz.py:89-143`)
   - All models plotted on same graph
   - AUC scores in legend
   - Random classifier baseline (diagonal line)

7. **Metrics Comparison** (`model_viz.py:145-203`)
   - Grouped bar chart comparing accuracy, precision, recall, F1
   - Value labels on bars
   - Allows quick visual comparison across models

8. **Comprehensive Dashboard** (`model_viz.py:313-474`)
   - Multi-panel figure with all visualizations
   - Summary table
   - Best model highlight box
   - Training time comparisons

### [DONE] Component 5: Data Mining Algorithms
**Location:** `src/models/sentiment_classifier.py`

**Three Algorithms Implemented:**

**1. Naive Bayes (MultinomialNB)** (Line 72)
- **Algorithm:** Probabilistic classifier based on Bayes' theorem
- **Why chosen:** Excellent baseline for text classification, fast training
- **Configuration:** Alpha=1.0 (Laplace smoothing)
- **Strengths:** Fast, works well with high-dimensional sparse data (text)
- **Typical performance:** High precision, good for interpretability

**2. Logistic Regression** (Lines 74-79)
- **Algorithm:** Linear model with sigmoid activation function
- **Why chosen:** Industry standard for binary classification, interpretable
- **Configuration:**
  - max_iter=1000 (sufficient for convergence)
  - C=1.0 (regularization strength)
  - solver='lbfgs' (efficient for medium datasets)
- **Strengths:** Balanced speed/accuracy, provides probability estimates
- **Typical performance:** Strong F1-score, reliable across datasets

**3. Random Forest** (Lines 81-88)
- **Algorithm:** Ensemble of 100 decision trees with majority voting
- **Why chosen:** Handles non-linear patterns, robust to overfitting
- **Configuration:**
  - n_estimators=100 (number of trees)
  - max_depth=50 (prevents overfitting)
  - min_samples_split=5 (requires 5+ samples to split node)
  - n_jobs=-1 (parallel processing using all CPU cores)
- **Strengths:** Often highest accuracy, captures complex relationships
- **Typical performance:** Best overall metrics but slower training

**Feature Engineering:**
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
  - Weights words by importance across documents
  - n-grams (1,2): captures single words and two-word phrases
  - max_features=5000: top 5000 most informative terms
  - min_df=2: word must appear in at least 2 documents
  - max_df=0.95: ignore words appearing in >95% of documents

**Evaluation Metrics:**
- Accuracy: Overall correctness
- Precision: Positive prediction reliability
- Recall: Coverage of actual positives
- F1-Score: Harmonic mean (primary selection metric)
- ROC-AUC: Discrimination ability across thresholds
- Training time & Prediction time: Efficiency metrics

### [DONE] Component 6: Actual Implementation

**A. Training Module** - `main.py`
- Loads Amazon reviews dataset
- Preprocesses 20,000 balanced samples
- Trains all 3 models with 80/20 train-test split
- Generates data exploration visualizations (saved to `plots/`)
- Compares model performance
- Saves best model to `artifacts/`
- Outputs detailed classification reports
- **Run with:** `python main.py`

**B. Deployment Module** - `app.py`
- Streamlit web application for model deployment
- Allows user to train models with custom parameters
- Provides model comparison dashboard
- Enables model selection for deployment
- Single text prediction with confidence scores
- Batch CSV prediction with downloadable results
- Real-time progress tracking during training
- **Run with:** `streamlit run app.py`

**Deployment Workflow:**
1. User uploads dataset in web interface
2. Configures training parameters (sample size, vectorization, etc.)
3. System trains all 3 models and displays comparison
4. User selects best model based on metrics
5. Deployed model used for predictions on new data
6. Results displayed in UI and downloadable as CSV

---

## Problem Domain: Detailed Description

### Business Context
Online retailers like Amazon receive millions of product reviews annually. Understanding customer sentiment at scale is critical for:

**Strategic Decision Making:**
- Product development priorities based on pain points
- Inventory management (stock popular items, discontinue poorly-rated ones)
- Marketing message optimization
- Competitive analysis

**Operational Efficiency:**
- Automated routing of negative reviews to customer service
- Quality assurance alerts for products with declining sentiment
- Fake review detection (when paired with other signals)

**Customer Experience:**
- Helping shoppers make informed purchase decisions
- Highlighting product strengths in search results
- Identifying and rewarding helpful reviewers

### Why Machine Learning?
Manual review analysis doesn't scale. A human can read ~50 reviews/hour. This system can classify 10,000+ reviews/minute after training, with consistent accuracy.

### Dataset Justification
Review datasets (e-commerce, hospitality, travel) are ideal for sentiment analysis because:
1. **Rich text data:** Customers write detailed, authentic opinions
2. **Ground truth labels:** Star ratings provide reliable sentiment labels
3. **High volume:** Sufficient data for training robust models
4. **Real-world applicability:** Directly useful across multiple industries
5. **Variety:** Multiple domains and rating scales test system flexibility
6. **Transferable insights:** Model trained on one domain can inform understanding of others

### Technical Challenges Addressed
1. **Imbalanced classes:** More 5-star than 1-star reviews → Solution: Class balancing
2. **Noisy text:** HTML, typos, abbreviations → Solution: Text preprocessing pipeline
3. **Variable length:** 10-1000+ characters → Solution: TF-IDF handles variable length
4. **Context dependence:** Sarcasm, negation → Solution: Bigrams capture some context
5. **Model selection:** Which algorithm performs best? → Solution: Train and compare all 3

---
