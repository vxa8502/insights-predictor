# Sentiment Analysis System - INSY 4325 Project

Amazon product review sentiment analysis using Machine Learning (Naive Bayes, Logistic Regression, Random Forest).

## Setup Instructions

### 1. Clone the Repository
Open your teminal app and type this:
```bash
git clone <your-github-repo-url>
cd SENTIMENT_ANALYTICS
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
### Download Dataset
The dataset is NOT in the repo (because it's too large). 

Download the Amazon Reviews dataset and place the CSV file here: `data/`. 

The link was shared on Teams a while ago. 

Make sure it is named `Reviews.csv`, that is what the training module: `main.py` expects. 

```
loader = AmazonDataLoader('data/Reviews.csv') 
```


### Dataset Info
- **Size:** ~500K reviews 
- **Columns Used:** Text, Score
- **Processing:** 1-2 stars → Negative, 4-5 stars → Positive (3-star reviews skipped)

## Usage

### Train Models
1. Go to "Train Models" page
2. Upload CSV with 'Text' and 'Score' columns
3. Click "Train Models"
4. View comparison and deploy best model

### Make Predictions
1. Go to "Make Predictions" page
2. Enter text or upload CSV
3. Click "Predict"
4. Download results


## Requirements
- Python 3.8+
- See `requirements.txt` for packages
