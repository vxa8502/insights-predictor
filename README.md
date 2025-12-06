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

**Download the dataset:**
1. Go to [Kaggle - Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)
2. Click the "Download" button (you may need to create a free Kaggle account)
3. The file will download directly to your Downloads folder as `Reviews.csv`

**Move the dataset to the project folder:**

For **Windows** (PowerShell):
```bash
# Move the CSV file from Downloads to the data folder
Move-Item -Path "$env:USERPROFILE\Downloads\Reviews.csv" -Destination "data\Reviews.csv"
```

For **Mac/Linux**:
```bash
# Move the CSV file from Downloads to the data folder (run this from the insights-predictor directory)
mv ~/Downloads/Reviews.csv data/Reviews.csv
```

**Verify the file is in the right place:**
```bash
# Windows
dir data\Reviews.csv

# Mac/Linux
ls -lh data/Reviews.csv
```

You should see the file listed with a size of approximately 280-300 MB.

### 6. Run Exploratory Data Analysis (Optional)
View raw data analysis before training:
```bash
# Run the EDA notebook
jupyter notebook exploratory_data_analysis.ipynb
```

**What to expect:**
- A browser window will automatically open at `http://localhost:8888/notebooks/exploratory_data_analysis.ipynb`
- You'll see the Jupyter notebook interface with code cells and markdown
- Click "Cell" → "Run All" from the menu to execute all cells
- The notebook will generate raw data visualizations and save them to the `plots/` folder
- When done, press `Ctrl+C` in the terminal to stop the Jupyter server

**Note:** The word frequency plots in this notebook show *raw unprocessed text*, so you'll see common stopwords like "the", "and", "is" dominating the charts. This is expected! The more interesting word frequencies and visualizations appear after text preprocessing when you run the training pipeline in step 7.

### 7. Train Models
Train all three ML models and generate performance visualizations:
```bash
python main.py
```

**What to expect:**
- **Duration:** ~2-3 minutes depending on your computer
- **Console output:** You'll see progress messages for:
  - Loading and preprocessing data (removing HTML, stopwords, etc.)
  - Class balancing (equalizing positive/negative reviews)
  - Training each model (Naive Bayes, Logistic Regression, Random Forest)
  - Generating performance metrics and visualizations
- **Generated files:**
  - `artifacts/` folder: Trained model files (.pkl) and vectorizer
  - `plots/` folder: Performance visualizations including:
    - Confusion matrices for each model
    - ROC curves comparison
    - Metrics comparison bar chart
    - Sentiment distribution
    - Text length distribution
    - Word clouds (positive/negative, after preprocessing)
    - Model dashboard summary
- **Final output:** Summary showing which model performed best (typically Random Forest or Logistic Regression)

**Verify success:** Open the `plots/` folder and check that visualizations were generated.

### 8. Launch the Web Application
Start the interactive prediction interface:
```bash
streamlit run app.py
```

**What to expect:**
- The app automatically opens in your browser at `http://localhost:8501`
- **Two main pages:**
  1. **Train Models:** Upload datasets, configure training, compare algorithms
  2. **Make Predictions:** Enter text or upload CSV for sentiment prediction
- **To stop:** Press `Ctrl+C` in the terminal

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

**Primary Dataset:** [Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)
- ~500,000 reviews across multiple product categories
- Columns: Text (review content), Score (1-5 star ratings)
- Download and place `Reviews.csv` in the `data/` folder (see step 5 above)
- **Note:** Dataset NOT included in repo due to size

### Multi-Dataset Support

The system automatically detects and works with various review dataset formats:
- **Text columns:** Tries 12+ common names (text, review, comment, content, etc.)
- **Rating columns:** Tries 11+ common names (rating, score, sentiment, etc.)
- **Rating scales:** Supports 1-5 stars, 1-10, 0-100, text labels (positive/negative), binary (0/1)

**Tested on:** Amazon, hotel, airline, restaurant, theme park, video game, and clothing reviews.

## Using the Web Application

After launching the app (step 8), you can:

**Train Models Page:**
1. Upload any review CSV dataset (auto-detects columns)
2. Configure sample size and vectorization method
3. Click "Train All Models" to compare Naive Bayes, Logistic Regression, and Random Forest
4. View performance metrics and visualizations
5. Select best model for deployment

**Make Predictions Page:**
1. Enter text directly or upload a CSV file
2. Click "Predict" to get sentiment classification
3. Download results if using batch predictions

## Advanced: Cross-Domain Analysis (Optional)

Test how models generalize across different review types:

1. Place multiple review CSV files in the `datasets/` folder
2. Run: `python cross_domain_analysis.py --sample-size 5000`
3. System trains on each domain and tests on all others
4. View results in `cross_domain_results.json` and `cross_domain_plots/`

See `FINAL_PROJECT_REPORT_PART1.md` for detailed cross-domain analysis results.

---
