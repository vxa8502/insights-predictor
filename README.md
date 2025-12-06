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

### Prerequisites (Windows)
- Terminal access: https://apps.microsoft.com/detail/9n0dx20hk701?hl=en-US&gl=US
- Python is installed: https://apps.microsoft.com/detail/9pnrbtzxmb4z?hl=en-US&gl=US
- Permission to run scripts:
  - Open PowerShell with Run as Administrator
  - Then, run command:
    ``` Set-ExecutionPolicy -ExecutionPolicy RemoteSigned```
  - Type Y and press Enter   

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

### 4. Download NLTK Data (One-time setup)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 5. Download Dataset
Create a `data/` folder and download the Amazon Reviews dataset:
```bash
mkdir data
```

Download `Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset) and place it in the `data/` folder as `data/Reviews.csv`.

### 6. Run Exploratory Data Analysis (Optional)
View raw data analysis before training:
```bash
# Install Jupyter if not already installed
pip install jupyter

# Run the EDA notebook
jupyter notebook exploratory_data_analysis.ipynb
```

Execute all cells to generate raw data visualizations in the `plots/` folder.

### 7. Verify Installation
Test the training pipeline:
```bash
python main.py
```

This should train models and generate plots in the `plots/` folder. Open this folder on your laptop and verify the visuals.

### 8. Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure
```
├── src/                                    # Core ML code
│   ├── data_loader.py                     # Dataset loading with class balancing
│   ├── text_preprocessor.py               # Text cleaning pipeline
│   ├── field_extractor.py                 # Flexible field detection
│   ├── models/sentiment_classifier.py     # ML model training
│   └── visualizations/                    # Visualization modules
│       ├── data_viz.py                    # Data exploration plots
│       └── model_viz.py                   # Model performance plots
├── data/                                   # Dataset folder (create this)
│   └── Reviews.csv                        # Amazon reviews (download separately)
├── datasets/                               # Cross-domain datasets (optional)
├── artifacts/                              # Trained models (auto-generated)
├── plots/                                  # Visualizations (included in submission)
├── exploratory_data_analysis.ipynb        # EDA notebook for raw data
├── cross_domain_analysis.py               # Cross-domain analysis tool (optional)
├── FINAL_PROJECT_REPORT_PART1.md          # Changes based on feedback
├── app.py                                  # Streamlit web interface
├── main.py                                 # Training script
└── requirements.txt                        # Python dependencies
```

## Dataset

### Primary Dataset (Used for Training)

**Amazon Product Reviews**
- **Source:** [Kaggle - Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)
- **Size:** ~500,000 reviews across multiple product categories
- **Columns:** Text (review content), Score (1-5 star ratings)
- **Download:** Place the `Reviews.csv` file in the `data/` folder
- **Note:** Dataset is NOT included in this repo due to size

**To use this dataset:**
```bash
mkdir data
# Download Reviews.csv from the Kaggle link above
# Place it in: data/Reviews.csv
```

### Compatible Dataset Formats

This system works with multiple review dataset formats through automatic field detection:

**Tested Datasets:**
- **Amazon Reviews** - Text/review columns, Score/rating columns
- **Hotel Reviews** - reviews.text, reviews.rating
- **Airline Reviews** - Review, Overall_Rating
- **Restaurant Reviews** - review/comment, rating/stars
- **Theme Park Reviews** - Review_Text, Rating

**How it works:**
- Automatically tries 12+ common text field names (text, review, comment, content, etc.)
- Automatically tries 11+ common rating field names (rating, score, sentiment, overall, etc.)
- Falls back to column position if names don't match

**Supported Rating Scales:**
- 1-5 stars: 1-2=negative, 3=neutral (skipped), 4-5=positive
- 1-10 scale: 1-4=negative, 5-6=neutral, 7-10=positive
- 0-100 percentage: 0-49=negative, 50-69=neutral, 70-100=positive
- Text labels: "positive", "negative", "pos", "neg"
- Binary: 0/1, True/False

### Dataset Validation

The system validates datasets before training and provides clear error messages:
- Checks text field contains actual text (not numeric IDs)
- Verifies rating field uses a supported format
- Ensures minimum data quality and quantity
- Shows helpful suggestions if dataset is incompatible

### Class Imbalance Handling

**The Problem:** Review datasets are heavily imbalanced - typically 80% positive, 10% negative, 10% neutral. This causes models to always predict "positive" and miss negative reviews.

**Our Solution:** Undersampling approach (`src/data_loader.py:389`)
1. Count positive and negative reviews after removing neutrals
2. Find the smaller class size (usually negative reviews)
3. Randomly sample equal amounts from both classes to create 50/50 balance
4. Uses `random_state=42` for reproducibility

**Why Undersampling?**
- We have enough data (500K+ reviews) to afford discarding excess positives
- Avoids creating duplicate examples (no overfitting)
- Faster training with balanced, smaller dataset
- Real data only - no synthetic examples

**Impact:** Without balancing, negative recall is ~12%. With balancing, negative recall jumps to ~86%. This ensures the model catches negative reviews effectively.

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


## Cross-Domain Analysis (Optional)

The `datasets/` folder supports cross-domain sentiment analysis experiments:

1. Add ~10 review CSV files from different domains (amazon, hotels, restaurants, etc.)
2. Run: `python cross_domain_analysis.py --sample-size 5000`
3. System trains on each domain and tests across all others
4. Results: `cross_domain_results.json` and plots in `cross_domain_plots/`

**Requirements:** Each CSV needs text/review and rating/sentiment columns (auto-detected). Supports different column names and rating scales.

## Requirements
- Python 3.8+
- See `requirements.txt` for packages

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
