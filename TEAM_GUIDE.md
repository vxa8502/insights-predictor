# Team Guide - Sentiment Analysis Project
## Quick Start Guide for Team Members

**Last Updated:** October 25, 2024

**Project Status:** Complete (excluding report & prez)

---

## Table of Contents
1. [Quick Setup (5 minutes)](#quick-setup)
2. [Project Overview](#project-overview)
3. [Running the System](#running-the-system)
4. [Code Architecture](#code-architecture)
5. [How Each Component Works](#how-each-component-works)
6. [How to Demo](#how-to-demo)
7. [Assignment Components Checklist](#assignment-components-checklist)
8. [Understanding the Code](#understanding-the-code)
9. [Quick Reference Commands](#quick-reference-commands)
10. [Performance Benchmarks](#performance-benchmarks)
11. [Presentation Q&A Prep](#presentation-qa-prep)
12. [Files to Not Commit](#files-to-not-commit)
13. [Additional Resources](#additional-resources)

---

## Quick Setup

### Prerequisites
- Python 3.8 or higher
- Terminal/Command Prompt access
- Amazon Reviews dataset (Reviews.csv) - 301MB file shared on Teams

### Step-by-Step Setup

**1. Clone/Download the Project**
```bash
cd ~/Desktop
cd sentiment_analytics
```

**2. Create Virtual Environment**
```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```
This takes 2-3 minutes. You'll see packages like pandas, scikit-learn, streamlit installing.

**4. Add Dataset**
- Download `Reviews.csv` from Teams
- Place it in the `data/` folder
- Path should be: `sentiment_analytics/data/Reviews.csv`

**5. Verify Setup**
```bash
# Test imports
python -c "import streamlit; import sklearn; import pandas; print('Setup successful!')"
```

If you see "Setup successful!" - you're ready!

---

## Project Overview

### What Does This System Do?

This is a **sentiment analysis system** that:
1. Takes Amazon product reviews (text)
2. Trains 3 machine learning models to classify sentiment
3. Deploys the best model through a web interface
4. Predicts if new reviews are POSITIVE or NEGATIVE

**Business Value:**
- Automates analysis of thousands of reviews
- Identifies unhappy customers quickly
- Provides data-driven insights for product improvement

### The Three Models

**1. Naive Bayes**
- Fast and simple
- Uses probability theory
- Good baseline performance

**2. Logistic Regression**
- Industry standard
- Balanced speed and accuracy
- Most reliable across different datasets

**3. Random Forest**
- Most complex
- Usually highest accuracy
- Takes longer to train

The system trains all 3 and lets you pick which one to deploy based on performance metrics.

## Running the System

### Method 1: Web Interface (RECOMMENDED for Demo)

**Start the app:**
```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

*What you'll see:*
- 5-page web application
- Home, Train Models, Make Predictions, Model Comparison, About

**To stop:**
Press `Ctrl+C` in terminal

### Method 2: Command Line (Faster for Training)

*Train models directly:*
```bash
python main.py
```

This will:
- Load 20,000 reviews from data/Reviews.csv
- Train all 3 models
- Generate plots in `plots/` folder
- Save best model to `artifacts/` folder
- Print performance comparison table

Takes 2-5 minutes depending on your computer.

## Code Architecture

### Project Structure
```
sentiment_analytics/
├── app.py                    # Web interface (Streamlit app)
├── main.py                   # Training script (command-line)
├── requirements.txt          # Python dependencies
├── data/
│   └── Reviews.csv          # Dataset (you add this)
├── src/                     # Core ML code
│   ├── data_loader.py       # Loads and cleans CSV data
│   ├── text_preprocessor.py # Cleans review text
│   ├── models/
│   │   └── sentiment_classifier.py  # 3 ML algorithms
│   └── visualizations/
│       ├── data_viz.py      # Data exploration charts
│       └── model_viz.py     # Model comparison charts
├── artifacts/               # Saved trained models (generated)
├── plots/                   # Charts and graphs (generated)
└── .streamlit/
    └── config.toml          # Streamlit settings
```

### Key Files Explained

**app.py** (800 lines)
- The web interface everyone will see during demo
- 5 pages: Home, Train, Predict, Compare, About
- Uses Streamlit framework
- **You'll demo this file**

**main.py** 
- Command-line training script
- Faster than web interface for large datasets
- Generates all plots automatically
- Good for testing changes quickly

**src/data_loader.py**
- Reads Reviews.csv file
- Converts 1-5 star ratings to positive/negative
- Balances dataset (equal positive/negative samples)
- Handles missing/invalid data

**src/text_preprocessor.py** 
- Cleans review text
- Removes HTML, punctuation, stop words
- Converts to lowercase
- Prepares text for ML models

**src/models/sentiment_classifier.py** 
- Contains all 3 ML algorithms
- Trains models
- Makes predictions
- Compares performance
- *The core of the project*

**src/visualizations/data_viz.py** 
- Creates charts about the dataset
- Sentiment distribution, word clouds, top words
- Review length analysis

**src/visualizations/model_viz.py**
- Creates charts comparing model performance
- Confusion matrices, ROC curves, metric comparisons
- Dashboard with all visualizations

## How Each Component Works

### 1. Data Loading Process
```python
# What happens when you load data:
loader = AmazonDataLoader('data/Reviews.csv')
reviews, stats = loader.load_data(sample_size=20000, balance=True)
```

*Steps:*
1. Reads CSV file (handles errors gracefully)
2. Converts scores: 1-2 stars = negative, 4-5 stars = positive, 3 stars = skip
3. Removes null/empty reviews
4. Balances classes (equal positive and negative)
5. Returns list of ReviewData objects

**Input:** Raw CSV with 500k reviews
**Output:** 20k balanced, clean reviews (10k positive, 10k negative)

### 2. Text Preprocessing
```python
preprocessor = TextPreprocessor()
clean_text = preprocessor.preprocess("This product is AMAZING! <html>Best buy!</html>")
# Returns: "product amazing best buy"
```

*What it does:*
- Removes HTML tags: `<html>` → removed
- Lowercase: `AMAZING` → `amazing`
- Removes punctuation: `!` → removed
- Removes stop words: `is`, `this` → removed
- Filters short words: words under 3 characters removed

**Why?** ML models work better with clean, normalized text.

### 3. Model Training
```python
analyzer = SentimentAnalyzer(vectorizer_type='tfidf', max_features=5000)
X_train, y_train = analyzer.prepare_data(texts, labels, fit_vectorizer=True)
results = analyzer.train_all_models(X_train, y_train, X_test, y_test)
```

*Steps:*
1. **Vectorization:** Converts text to numbers using TF-IDF
   - "great product" → [0.0, 0.8, 0.0, 0.6, ...] (5000 numbers)
2. **Training:** Trains all 3 models simultaneously
3. **Evaluation:** Tests on held-out 20% of data
4. **Comparison:** Calculates accuracy, precision, recall, F1, ROC-AUC

**Output:** Dictionary with results for each model

### 4. Making Predictions
```python
# Single prediction
predictions = analyzer.predict(["This product is terrible"])
# Returns: [{'sentiment': 'negative', 'confidence': 0.92, 'model': 'Random Forest'}]
```

**How it works:**
1. Preprocess the input text
2. Convert to numbers (vectorization)
3. Pass through trained model
4. Get probability scores
5. Return sentiment + confidence


## How to Demo
**1. Introduction (30 seconds)**
"Our project is a sentiment analysis system for Amazon product reviews. It automatically classifies whether reviews are positive or negative using machine learning."

**2. Show the Web App (30 seconds)**
```bash
streamlit run app.py
```
- Navigate through the 5 pages quickly
- "This is our deployment interface where users can train models and make predictions"

**3. Train Models (2 minutes)**
- Go to "Train Models" page
- Upload Reviews.csv
- Set sample size to 5,000 (faster demo)
- Click "Train All Models"
- Show progress bar
- Point out real-time status updates

**4. Show Results (2 minutes)**
- Performance comparison table appears
- "Here we can see all 3 models trained and compared"
- Point out accuracy, F1-score, ROC-AUC
- Show confusion matrices visualization
- Show ROC curves
- "Random Forest usually performs best with ~90% accuracy"

**5. Make Predictions (1 minute)**
- Deploy the best model
- Go to "Make Predictions" page
- Type example: "This product is amazing! Best purchase ever."
- Click predict
- Show result: POSITIVE with 95% confidence
- Try negative: "Terrible quality, waste of money"
- Show result: NEGATIVE with 89% confidence

**6. Model Comparison Page (1 minute)**
- Navigate to "Model Comparison"
- Show detailed metrics for all models
- Point out visualizations
- "This helps us make an informed decision about which model to deploy"

**7. Wrap Up (30 seconds)**
- "The system is fully functional and ready for production use"
- "It can handle batch predictions via CSV upload"
- "All results are downloadable"

### Demo Tips

**DO:**
- Practice the demo 2-3 times before presentation
- Use smaller sample size (5,000) for faster training during demo
- Have backup screenshots in case of technical issues
- Prepare example review texts in advance
- Know the performance metrics (accuracy ~90%)

**DON'T:**
- Don't train with full 20,000 samples during demo (takes too long)
- Don't upload the full 301MB dataset (use 5-10k sample)
- Don't skip the visualizations (they're impressive!)
- Don't forget to activate venv before demo

### Backup Plan
If live demo fails:
1. Show pre-generated plots from `plots/` folder
2. Run `main.py` beforehand and show terminal output
3. Have screenshots of the web app ready
4. Walk through the code instead

---

## Assignment Components Checklist

Use this to verify everything is complete:

### Component 1: Problem Domain Description
- **Location:** README.md (lines 357-395) + app.py "About" page
- **What to say:** "We analyzed Amazon product reviews to automatically determine customer sentiment. This helps businesses process thousands of reviews quickly."
- **Status:** COMPLETE

### Component 2: User Interface Layout/Design
- **Location:** app.py (the web interface) + UI_DESIGN.md
- **What to show:** Live demo of the 5-page Streamlit app
- **Status:** COMPLETE

### Component 3: Data Cleaning Techniques
- **Location:** src/data_loader.py + src/text_preprocessor.py
- **What to explain:**
  - Score-to-sentiment mapping (1-2 = negative, 4-5 = positive)
  - HTML removal
  - Text normalization (lowercase, punctuation)
  - Stop word removal
  - Class balancing
- **Status:** COMPLETE (7 techniques implemented)

### Component 4: Visualization Techniques
- **Location:** src/visualizations/data_viz.py + model_viz.py
- **What to show:** plots/ folder with 8 different charts
  - Sentiment distribution
  - Word clouds
  - Review length analysis
  - Top words
  - Confusion matrices
  - ROC curves
  - Metrics comparison
  - Comprehensive dashboard
- **Status:** COMPLETE (8 visualizations)

### Component 5: Data Mining Algorithms
- **Location:** src/models/sentiment_classifier.py
- **What to explain:**
  - Naive Bayes (probabilistic, fast)
  - Logistic Regression (linear, reliable)
  - Random Forest (ensemble, most accurate)
- **Why these 3:** Cover different approaches (probabilistic, linear, ensemble)
- **Status:** COMPLETE

### Component 6: Actual Implementation
- **Training Module:** main.py
- **Deployment Module:** app.py
- **What to demonstrate:**
  - Train all 3 models (live demo or show output)
  - Compare performance
  - Deploy best model
  - Make predictions on new data
- **Status:** COMPLETE

## Understanding the Code

**Step 1: Preprocessing**
```
Input:  "This product is AMAZING! I love it <3"
Output: "product amazing love"
```

**Step 2: Vectorization (TF-IDF)**
```
Vocabulary: [amazing, bad, love, product, terrible, ...]
"product amazing love" → [0.8, 0.0, 0.7, 0.9, 0.0, ...]
```
Each word gets a number based on how important it is.

**Step 3: Machine Learning**
The model learns patterns like:
- Words like "amazing", "love", "great" → POSITIVE
- Words like "terrible", "bad", "waste" → NEGATIVE

### Why Three Models?

**Different strengths:**
- **Naive Bayes:** Fast, good for large datasets
- **Logistic Regression:** Balanced, most reliable
- **Random Forest:** Complex, highest accuracy

**We train all 3 and pick the best** based on F1-score (balance of precision and recall).

### What's F1-Score?

**Simple explanation:**
- **Accuracy:** Overall correctness (90% = 9 out of 10 right)
- **Precision:** When we say POSITIVE, how often are we right?
- **Recall:** Of all actual POSITIVE reviews, how many did we find?
- **F1-Score:** Combines precision and recall into one number

**For our demo:** Focus on accuracy and F1-score (both around 0.90 = 90%)

---


## Quick Reference Commands

### Setup
```bash
cd ~/Desktop/sentiment_analytics
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### Run Web App
```bash
streamlit run app.py
```

### Train Models (Command Line)
```bash
python main.py
```

### Check Everything Works
```bash
python -c "import streamlit, sklearn, pandas; print('OK')"
```

### View Generated Plots
```bash
# Mac
open plots/

# Windows
explorer plots\
```

### Stop Running App
Press `Ctrl+C` in terminal

---

## Performance Benchmarks

**Expected Results (20,000 samples):**

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| Naive Bayes | ~87.6% | ~0.875 | 2-3 seconds |
| Logistic Regression | ~89.1% | ~0.890 | 5-7 seconds |
| Random Forest | ~90.3% | ~0.902 | 40-50 seconds |

**With 5,000 samples (for faster demo):**
- Accuracy slightly lower (~85-88%)
- Training much faster (10-15 seconds total)
- Still impressive for demo!

---

## Presentation Q&A Prep

**Q: Why did you choose these three algorithms?**
A: "We wanted to compare different approaches: probabilistic (Naive Bayes), linear (Logistic Regression), and ensemble (Random Forest). This gives us the best chance of finding the optimal model for our specific dataset."

**Q: How accurate is your system?**
A: "Our best model (Random Forest) achieves 90% accuracy and an F1-score of 0.902. This means it correctly classifies 9 out of 10 reviews."

**Q: How does the system handle sarcasm?**
A: "Currently, the system struggles with sarcasm like most text-based sentiment analysis. We use bigrams (two-word phrases) which helps capture some context, but advanced sarcasm detection would require more sophisticated NLP techniques like contextual embeddings."

**Q: What preprocessing did you do?**
A: "We implemented 7 cleaning techniques: HTML removal, text normalization, stop word filtering, score mapping, class balancing, null handling, and short word filtering. This ensures the ML models receive clean, consistent input."

**Q: Why exclude 3-star reviews?**
A: "3-star reviews are neutral and ambiguous - they make classification harder. By focusing on clearly positive (4-5 stars) and negative (1-2 stars) reviews, we get better model performance and clearer business insights."

**Q: How do you deploy the model?**
A: "We built a Streamlit web application where users can upload datasets, train models, compare performance, select the best model, and make predictions on new reviews. The deployed model is saved and can be loaded for future use."

**Q: What's the business value?**
A: "Businesses receive thousands of reviews daily. Manual analysis is slow (50 reviews/hour). Our system processes 10,000+ reviews per minute with 90% accuracy, enabling quick identification of customer issues and data-driven decisions."

**Q: Could this work for other products or industries?**
A: "Yes! While we trained on Amazon product reviews, the system could be retrained on restaurant reviews, movie reviews, customer service feedback, or any text-based sentiment data."

---

## Files to Not Commit

Before commiting, make sure these files are not in the repo:

**DON'T COMMIT:**
- [ ] venv/ (virtual environment - too large)
- [ ] data/Reviews.csv (301MB - share separately)
- [ ] __pycache__/ (Python cache)
- [ ] .DS_Store (Mac system files)
- [ ] artifacts/*.pkl (trained models - optional, can regenerate)
- [ ] plots/*.png (visualizations - optional, can regenerate)

The `.gitignore` file already handles most of these.

---
## Additional Resources

*Learn more:*

1. **Streamlit Documentation:**
   https://docs.streamlit.io/

2. **Scikit-learn User Guide:**
   https://scikit-learn.org/stable/user_guide.html

3. **Understanding TF-IDF:**
   https://en.wikipedia.org/wiki/Tf%E2%80%93idf

4. **Sentiment Analysis Overview:**
   https://en.wikipedia.org/wiki/Sentiment_analysis

5. **Project README:**
   See README.md for detailed component mapping

6. **UI Design:**
   See UI_DESIGN.md for interface documentation

---

**Questions?** Review this guide, check README.md, or run the code!
