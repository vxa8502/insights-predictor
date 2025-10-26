# Sentiment Analysis System - Design Document

## INSY 4325 Business Analytics Project - Option 2

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
[User Interface Layer]
        |
        v
[Application Layer]
        |
        v
[Data Processing Layer]
        |
        v
[Model Layer]
        |
        v
[Storage Layer]
```

### 1.2 Component Architecture

**Frontend:**
- Streamlit Web Application (app.py)
- Responsive multi-page interface
- Real-time progress tracking
- Interactive visualizations

**Backend:**
- Python-based machine learning pipeline
- Modular component design
- Artifact persistence system
- Logging and error handling

**Data Layer:**
- CSV file ingestion
- Flexible field detection system
- Multi-format rating normalization
- Preprocessed data caching

**Model Layer:**
- Three parallel ML models
- Unified prediction interface
- Model serialization and versioning
- Performance comparison engine

---

## 2. User Interface Design

### 2.1 Navigation Structure

```
Main Application
|
+-- Home (Dashboard)
|   +-- System status indicators
|   +-- Quick start guide
|   +-- Dataset requirements
|
+-- Train Models
|   +-- File upload interface
|   +-- Training configuration panel
|   +-- Progress tracking
|   +-- Results visualization
|   +-- Model deployment selector
|
+-- Make Predictions
|   +-- Model status display
|   +-- Single text prediction mode
|   +-- Batch CSV prediction mode
|   +-- Results export
|
+-- Model Comparison
|   +-- Performance metrics table
|   +-- Detailed model cards
|   +-- Visualization gallery
|   +-- Best model recommendation
|
+-- About
    +-- Project documentation
    +-- System information
    +-- Assignment mapping
```

### 2.2 Page-Level Design Specifications

#### **Page 1: Home Dashboard**

**Purpose:** Entry point providing system overview and status

**Layout:**
```
+---------------------------------------+
|          HEADER: Title                |
+---------------------------------------+
|                                       |
|    Welcome Message & Description      |
|                                       |
+---------------------------------------+
|    How to Use (3-step guide)          |
+---------------------------------------+
|    Dataset Requirements               |
+---------------------------------------+
|  [Status 1] | [Status 2] | [Status 3] |
+---------------------------------------+
```

**Components:**
- Main header: Large centered title with blue color scheme
- Instructional text: Clear numbered steps
- Status indicators: Three columns showing:
  - Training status (trained/not trained)
  - Deployed model name
  - Saved artifacts count
- Color coding: Green for success, Blue for info, Red for warnings

**User Flow:**
1. User lands on home page
2. Reads instructions
3. Checks system status
4. Navigates to Train Models or Make Predictions

---

#### **Page 2: Train Models**

**Purpose:** Configure and train all three ML models

**Layout:**
```
+---------------------------------------+
|          HEADER: Train Models         |
+---------------------------------------+
|    Step 1: Upload Dataset             |
|    [File Upload Widget]               |
+---------------------------------------+
|    Step 2: Configure Parameters       |
|    [Sample Size Slider]               |
|    [Balance Classes Checkbox]         |
|    [Vectorization Dropdown]           |
|    [Max Features Slider]              |
+---------------------------------------+
|    Step 3: Train                      |
|    [Train All Models Button]          |
+---------------------------------------+
|    [Progress Bar]                     |
|    [Status Text]                      |
+---------------------------------------+
|    Performance Comparison Table       |
+---------------------------------------+
| [Confusion Matrix] | [Metrics Chart]  |
+---------------------------------------+
|    [ROC Curves]                       |
+---------------------------------------+
|    Step 4: Deploy Model               |
|    [Model Selection Dropdown]         |
|    [Deploy Button]                    |
+---------------------------------------+
```

**Components:**

1. **File Upload Widget**
   - Type: CSV only
   - Help text: Column requirements
   - Validation: Displays extracted field information

2. **Configuration Panel**
   - Sample Size: Number input (1,000 - 100,000)
   - Balance Classes: Checkbox toggle
   - Vectorization Method: Dropdown (TF-IDF, Count)
   - Max Features: Slider (1,000 - 10,000)

3. **Progress Tracking**
   - Progress bar: 0-100% visual indicator
   - Status text: Current operation description
   - Real-time updates during training

4. **Results Display**
   - Comparison table: All metrics side-by-side
   - Visualization grid: 2-column layout
   - Image displays: Confusion matrices, ROC curves, metric bars

5. **Deployment Interface**
   - Model selector: Dropdown pre-populated with best model
   - Deploy button: Primary action button
   - Confirmation message: Success feedback with navigation prompt

**User Flow:**
1. Upload CSV dataset
2. System auto-detects fields and displays info
3. Adjust training parameters (or use defaults)
4. Click "Train All Models"
5. Monitor progress bar and status updates
6. Review performance comparison
7. Select best model for deployment
8. Navigate to predictions page

**Design Decisions:**
- Step-by-step numbered layout for clarity
- Pre-filled defaults for quick start
- Visual progress indicators for long operations
- Immediate feedback on file upload
- Best model pre-selected in deployment dropdown

---

#### **Page 3: Make Predictions**

**Purpose:** Use deployed model to predict sentiment on new data

**Layout:**
```
+---------------------------------------+
|       HEADER: Make Predictions        |
+---------------------------------------+
|    [Model Status Banner]              |
+---------------------------------------+
|    Mode: ( ) Single | ( ) Batch       |
+---------------------------------------+
|                                       |
|    === SINGLE TEXT MODE ===           |
|    [Text Area Input]                  |
|    [Predict Button]                   |
|                                       |
|    Result Display:                    |
|    [Sentiment] | [Confidence] | [Model]|
|    [Confidence Progress Bar]          |
|                                       |
|    --- OR ---                         |
|                                       |
|    === BATCH CSV MODE ===             |
|    [File Upload Widget]               |
|    [Data Preview Table]               |
|    [Predict All Button]               |
|                                       |
|    Results:                           |
|    [Full Results Table]               |
|    [Pos Count] | [Neg Count] | [Avg]  |
|    [Download CSV Button]              |
|                                       |
+---------------------------------------+
```

**Components:**

1. **Model Status Banner**
   - Success indicator: Green banner showing deployed model
   - Warning indicator: Yellow banner if no model deployed
   - Action prompt: Link to training page if needed

2. **Mode Selection**
   - Radio buttons: Single Text / Batch CSV
   - Horizontal layout for clear choice
   - Switches interface below

3. **Single Text Interface**
   - Text area: 150px height, placeholder example
   - Predict button: Full-width primary action
   - Result cards: 3-column metric display
   - Confidence bar: Visual progress indicator
   - Color coding: Green for positive, Red for negative

4. **Batch CSV Interface**
   - File uploader: CSV only
   - Preview table: First 5 rows displayed
   - Predict button: Full-width primary action
   - Results table: All predictions with scrolling
   - Summary metrics: 3-column statistics
   - Download button: CSV export functionality

**User Flow - Single Text:**
1. Select "Single Text" mode
2. Enter or paste review text
3. Click "Predict Sentiment"
4. View sentiment, confidence, and model name
5. Interpret confidence visualization

**User Flow - Batch CSV:**
1. Select "Batch Upload" mode
2. Upload CSV file with 'Text' column
3. Review data preview
4. Click "Predict All"
5. View results table and summary statistics
6. Download results CSV

**Design Decisions:**
- Mode toggle at top for clear distinction
- Separate interfaces to avoid confusion
- Immediate preview of uploaded batch data
- Summary statistics for batch results
- Downloadable results for business use

---

#### **Page 4: Model Comparison**

**Purpose:** Detailed analysis of all trained models

**Layout:**
```
+---------------------------------------+
|      HEADER: Model Comparison         |
+---------------------------------------+
|    Performance Metrics Table          |
|    (All models, all metrics)          |
+---------------------------------------+
|    Detailed Model Metrics             |
|                                       |
|    > [Model 1 Expandable]             |
|      [5-column metric cards]          |
|      [2-column time metrics]          |
|                                       |
|    > [Model 2 Expandable]             |
|      [metrics...]                     |
|                                       |
|    > [Model 3 Expandable]             |
|      [metrics...]                     |
+---------------------------------------+
|    Performance Visualizations         |
|                                       |
|    [Confusion Matrices Image]         |
|                                       |
| [Metrics Comparison] | [ROC Curves]   |
|                                       |
|    [Comprehensive Dashboard]          |
+---------------------------------------+
|    Recommendation Box                 |
|    (Best model with justification)    |
+---------------------------------------+
```

**Components:**

1. **Comparison Table**
   - Full-width dataframe display
   - All metrics for all models
   - Sortable columns
   - Highlighted best values

2. **Expandable Model Cards**
   - Collapsed by default for clean interface
   - 5-column metric display: Accuracy, Precision, Recall, F1, ROC-AUC
   - 2-column time metrics: Training time, Prediction time
   - Consistent layout across all models

3. **Visualization Gallery**
   - Full-width confusion matrix image
   - 2-column layout for comparison charts
   - Comprehensive dashboard image
   - All images with descriptive captions

4. **Recommendation Panel**
   - Info-styled box (blue background)
   - Bold model name
   - Justification based on F1-score
   - Key metrics listed

**Design Decisions:**
- Table first for quick comparison
- Expandable cards reduce clutter
- Multiple visualization formats
- Clear recommendation with reasoning

---

#### **Page 5: About**

**Purpose:** Project documentation and system information

**Layout:**
```
+---------------------------------------+
|          HEADER: About                |
+---------------------------------------+
|    Project Overview                   |
|    (Course, assignment type)          |
+---------------------------------------+
|    System Information                 |
|                                       |
|  [Directories Info] | [Session Status]|
+---------------------------------------+
```

**Components:**
- Markdown documentation
- 2-column info boxes
- Directory status checks
- Session state display

---

### 2.3 Design System

#### **Color Palette**

**Primary Colors:**
- Blue (#1f77b4): Headers, primary actions, info messages
- Green (#2ca02c): Positive sentiment, success indicators
- Red (#d62728): Negative sentiment, error messages
- Gray (#7f7f7f): Secondary text, borders

**Background Colors:**
- Light Gray (#f0f2f6): Metric cards, containers
- White (#ffffff): Main background
- Light Blue (info boxes): System messages

#### **Typography**

**Headers:**
- Main header: 2.5rem, bold, blue, center-aligned
- Section headers: 1.5rem, bold, default color
- Subsection headers: 1.2rem, bold

**Body Text:**
- Standard: 1rem, regular weight
- Captions: 0.9rem, gray color
- Metrics: 1.2rem, bold

#### **Component Styles**

**Buttons:**
- Primary: Blue background, white text, full-width option
- Secondary: Gray background, dark text
- Hover state: Slightly darker shade
- Disabled state: Light gray, reduced opacity

**Input Fields:**
- Border: 1px gray
- Focus: Blue border
- Placeholder: Light gray text
- Height: Appropriate to content type

**Cards/Containers:**
- Background: Light gray (#f0f2f6)
- Padding: 1rem
- Border radius: 0.5rem
- Margin: 0.5rem vertical spacing

**Progress Indicators:**
- Progress bar: Blue fill, gray background
- Spinner: Centered, blue color
- Status text: Below bar, italic

---

### 2.4 Responsive Design Considerations

**Column Layouts:**
- Desktop: Multi-column layouts (2-3 columns)
- Tablet: Reduced to 2 columns
- Mobile: Single column stack

**Image Scaling:**
- Full-width option for large visualizations
- Column-width for side-by-side comparisons
- Automatic scaling on smaller screens

**Navigation:**
- Sidebar: Collapsible on mobile
- Radio buttons: Horizontal on desktop, stack on mobile
- Tables: Horizontal scrolling on mobile

---

## 3. Data Flow Design

### 3.1 Training Pipeline Data Flow

```
CSV File Upload
      |
      v
Field Detection & Validation
      |
      v
Data Loading (sample_size, balance)
      |
      v
Rating Normalization
      |
      v
Text Preprocessing
      |
      v
Data Splitting (80/20)
      |
      v
Text Vectorization (TF-IDF)
      |
      v
Parallel Model Training
      |
      +-- Naive Bayes
      +-- Logistic Regression
      +-- Random Forest
      |
      v
Model Evaluation
      |
      v
Performance Comparison
      |
      v
Visualization Generation
      |
      v
Model Serialization
      |
      v
Deployment Selection
```

**Data Transformations:**

1. **Field Detection:**
   - Input: Raw CSV DataFrame
   - Process: Try 12+ text field names, 11+ rating field names
   - Output: Validated text and sentiment columns
   - Error handling: Clear messages for incompatible datasets

2. **Rating Normalization:**
   - Input: Raw rating values (various scales/formats)
   - Process: Map to binary positive/negative
   - Output: Cleaned sentiment labels
   - Neutral handling: Excluded from dataset

3. **Text Preprocessing:**
   - Input: Raw review text
   - Process: HTML removal, lowercase, punctuation removal, stop words, short words
   - Output: Cleaned token list
   - Preservation: Sentiment-bearing words like "not", "but"

4. **Vectorization:**
   - Input: Preprocessed text strings
   - Process: TF-IDF transformation with bigrams
   - Output: Sparse feature matrix
   - Configuration: 5000 features, min_df=2, max_df=0.95

---

### 3.2 Prediction Pipeline Data Flow

```
User Input (Text or CSV)
      |
      v
Text Preprocessing
      |
      v
Vectorization (using trained vectorizer)
      |
      v
Model Prediction
      |
      v
Confidence Score Calculation
      |
      v
Result Formatting
      |
      v
Display/Export
```

**Single Text Flow:**
- Input: Text string
- Processing: Same pipeline as training
- Output: Sentiment label, confidence percentage, model name
- Display: Metric cards and confidence bar

**Batch CSV Flow:**
- Input: CSV with 'Text' column
- Processing: Vectorized prediction for all rows
- Output: DataFrame with added columns (sentiment, confidence, model)
- Export: Downloadable CSV file

---

### 3.3 Model Management Flow

```
Training Complete
      |
      v
Model Serialization (pickle)
      |
      v
Save to artifacts/ directory
      |
      v
Session State Storage
      |
      v
Model Selection Interface
      |
      v
Deployment (set as active)
      |
      v
Available for Predictions
```

---

## 4. Component Design Specifications

### 4.1 Data Loader Component

**Responsibility:** Load and validate datasets from multiple formats

**Class:** `DataLoader` (src/data_loader.py)

**Key Methods:**
- `load_data(sample_size, balance)`: Main loading function
- `_detect_and_validate_fields()`: Flexible field detection
- `_validate_dataset()`: Data quality checks
- `_normalize_sentiment()`: Multi-scale rating handling
- `_balance_classes()`: Stratified sampling

**Design Patterns:**
- Strategy pattern for rating normalization
- Template method for validation
- Defensive programming with detailed error messages

**Configuration:**
- Supports 5+ dataset types
- 12+ text field name variations
- 11+ rating field name variations
- Multiple rating scale formats

---

### 4.2 Text Preprocessor Component

**Responsibility:** Clean and normalize text data

**Class:** `TextPreprocessor` (src/text_preprocessor.py)

**Key Methods:**
- `preprocess(text)`: Main preprocessing pipeline
- `_remove_html(text)`: Strip HTML tags
- `_remove_punctuation(text)`: Clean special characters
- `_remove_stopwords(tokens)`: Filter non-informative words

**Pipeline Stages:**
1. HTML removal (regex-based)
2. Lowercase conversion
3. Punctuation removal (preserve apostrophes)
4. Tokenization
5. Stop word filtering (custom list)
6. Short word removal (< 3 chars)

**Configuration:**
- Custom stop word list (60+ words)
- Preserves sentiment-critical words
- Optional lemmatization (disabled by default)

---

### 4.3 Sentiment Analyzer Component

**Responsibility:** Train, evaluate, and deploy ML models

**Class:** `SentimentAnalyzer` (src/models/sentiment_classifier.py)

**Key Methods:**
- `prepare_data()`: Vectorization
- `train_all_models()`: Parallel training
- `predict()`: Sentiment prediction
- `get_best_model()`: Model selection
- `save_model()` / `load_model()`: Persistence

**Model Configurations:**

1. **Naive Bayes:**
   - Algorithm: MultinomialNB
   - Parameters: alpha=1.0
   - Strengths: Fast, interpretable

2. **Logistic Regression:**
   - Algorithm: LogisticRegression
   - Parameters: max_iter=1000, C=1.0, solver='lbfgs'
   - Strengths: Balanced performance

3. **Random Forest:**
   - Algorithm: RandomForestClassifier
   - Parameters: n_estimators=100, max_depth=50, min_samples_split=5
   - Strengths: High accuracy

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Training time, Prediction time
- Confusion matrix values

---

### 4.4 Visualization Components

**Responsibility:** Generate data exploration and model performance visualizations

**Classes:**
- `DataVisualizer` (src/visualizations/data_viz.py)
- `ModelVisualizer` (src/visualizations/model_viz.py)

**Data Visualizations:**
1. Sentiment distribution bar chart
2. Review length box plots and histograms
3. Word clouds (positive/negative)
4. Top words bar charts

**Model Visualizations:**
1. Confusion matrices (heatmaps)
2. ROC curves (all models overlaid)
3. Metrics comparison (grouped bar chart)
4. Comprehensive dashboard (multi-panel figure)

**Design Standards:**
- Consistent color schemes
- Professional fonts (Arial, sans-serif)
- Annotated values on charts
- High-resolution exports (300 DPI)
- Descriptive titles and labels

---

## 5. Database/Storage Design

### 5.1 File System Structure

```
sentiment_analytics/
|
+-- data/
|   +-- Reviews.csv              (input dataset)
|   +-- temp_upload.csv          (uploaded file cache)
|   +-- processed/               (optional preprocessed data)
|
+-- artifacts/
|   +-- naive_bayes_model.pkl
|   +-- logistic_regression_model.pkl
|   +-- random_forest_model.pkl
|
+-- plots/
|   +-- sentiment_dist.png
|   +-- length_dist.png
|   +-- wordcloud_positive.png
|   +-- wordcloud_negative.png
|   +-- top_words.png
|   +-- confusion_matrices.png
|   +-- roc_curves.png
|   +-- metrics_comparison.png
|   +-- model_dashboard.png
|
+-- src/
|   +-- data_loader.py
|   +-- text_preprocessor.py
|   +-- field_extractor.py
|   +-- models/
|   |   +-- sentiment_classifier.py
|   +-- visualizations/
|       +-- data_viz.py
|       +-- model_viz.py
|
+-- app.py                       (Streamlit application)
+-- main.py                      (training script)
+-- requirements.txt
+-- README.md
+-- DESIGN.md                    (this document)
```

### 5.2 Artifact Storage

**Model Serialization:**
- Format: Pickle (.pkl)
- Contents: Trained sklearn model + vectorizer
- Naming: `{model_name}_model.pkl`
- Location: `artifacts/` directory

**Model Object Structure:**
```python
{
    'vectorizer': TfidfVectorizer,  # Fitted vectorizer
    'models': {
        'Naive Bayes': MultinomialNB,
        'Logistic Regression': LogisticRegression,
        'Random Forest': RandomForestClassifier
    },
    'metadata': {
        'trained_date': datetime,
        'sample_size': int,
        'vectorizer_config': dict,
        'performance': dict
    }
}
```

### 5.3 Session State Management

**Streamlit Session State Variables:**
- `analyzer`: SentimentAnalyzer instance
- `preprocessor`: TextPreprocessor instance
- `trained`: Boolean flag
- `results`: Dictionary of ModelResult objects
- `selected_model`: String model name

**Persistence Strategy:**
- In-memory during session
- Disk-based for model artifacts
- Temporary files for uploads
- No database required

---

## 6. Performance Design

### 6.1 Processing Optimization

**Data Loading:**
- Chunked reading for large files
- Early validation to fail fast
- Stratified sampling for balance
- Capped at 50k samples per class

**Preprocessing:**
- Vectorized operations where possible
- Regex compilation reuse
- Stop word set (O(1) lookup)
- Minimal string operations

**Model Training:**
- Parallel processing for Random Forest
- Optimized solver selection
- Limited max iterations
- Early stopping where applicable

### 6.2 Expected Performance Metrics

**Training Time:**
- Naive Bayes: 1-3 seconds
- Logistic Regression: 5-15 seconds
- Random Forest: 30-90 seconds

**Prediction Time:**
- Single text: < 100ms
- Batch (1000 reviews): 1-3 seconds

**Memory Usage:**
- Training: 500MB - 2GB
- Prediction: 200MB - 500MB
- Artifacts: 50-200MB per model

### 6.3 Scalability Considerations

**Current Limits:**
- Max sample size: 100,000 reviews
- Max features: 10,000
- Max batch prediction: No hard limit (memory-dependent)

**Scaling Strategies:**
- Incremental learning for larger datasets
- Feature selection for dimensionality reduction
- Model compression for deployment
- Caching for repeated predictions

---

## 7. Error Handling Design

### 7.1 Validation Strategy

**Input Validation:**
- File format checking (CSV only)
- Column existence verification
- Data type validation
- Minimum sample size enforcement

**Data Quality Checks:**
- Text field contains actual text (not IDs)
- Rating field uses supported format
- Sufficient non-null values
- Minimum average text length

**Error Message Design:**
- Clear description of the problem
- Specific field/value that failed
- Suggestion for resolution
- Example of correct format

### 7.2 Error Handling Hierarchy

**Level 1: User-Friendly Messages**
- Displayed in Streamlit UI
- Use st.error(), st.warning(), st.info()
- Provide actionable guidance
- No technical stack traces

**Level 2: Logging**
- Console logging for debugging
- Log levels: INFO, WARNING, ERROR
- Include context and timestamps
- Exception details in logs only

**Level 3: Fallback Behavior**
- Graceful degradation when possible
- Default parameters for missing values
- Column position fallback for field detection
- Skip neutral ratings instead of failing

---

## 8. Security and Privacy Design

### 8.1 Data Security

**File Handling:**
- Temporary upload files deleted after processing
- No persistent storage of user data
- Local file system only (no cloud)

**Model Security:**
- Read-only model loading
- Validated pickle deserialization
- No executable code in models

### 8.2 Privacy Considerations

**Data Privacy:**
- All processing happens locally
- No external API calls
- No telemetry or tracking
- User data never leaves the application

**Input Sanitization:**
- HTML stripping in preprocessing
- No code execution from input
- Safe regex patterns
- Validated file uploads

---

## 9. Testing Strategy

### 9.1 Unit Testing (Recommended)

**Data Loader Tests:**
- Field detection accuracy
- Rating normalization correctness
- Validation error handling
- Balance functionality

**Preprocessor Tests:**
- HTML removal
- Stop word filtering
- Token length requirements
- Pipeline integration

**Model Tests:**
- Training completion
- Prediction output format
- Model serialization/deserialization
- Performance metric calculation

### 9.2 Integration Testing

**End-to-End Workflows:**
- Complete training pipeline
- Single text prediction
- Batch CSV prediction
- Model switching

**UI Testing:**
- Page navigation
- File upload handling
- Progress tracking
- Result display

### 9.3 Validation Testing

**Dataset Compatibility:**
- Test with 5+ different datasets
- Various rating scales
- Different column names
- Edge cases (small datasets, imbalanced)

**Model Performance:**
- Baseline accuracy thresholds
- Consistency across runs
- Comparison with expectations

---

## 10. Deployment Considerations

### 10.1 Local Deployment

**Requirements:**
- Python 3.8+
- Virtual environment recommended
- Dependencies from requirements.txt
- 2GB+ RAM available

**Installation:**
1. Clone repository
2. Create virtual environment
3. Install dependencies
4. Download dataset
5. Run training or launch app

### 10.2 Streamlit Deployment

**Streamlit Cloud:**
- Compatible with Streamlit sharing
- Requires GitHub repository
- Resource limits apply
- Large dataset may exceed limits

**Configuration:**
- `.streamlit/config.toml` for settings
- Secrets management for API keys (if needed)
- Custom theme configuration

### 10.3 Production Considerations (Future)

**API Deployment:**
- FastAPI wrapper for model
- REST endpoints for prediction
- Docker containerization
- Load balancing for scale

**Monitoring:**
- Prediction logging
- Performance metrics tracking
- Error rate monitoring
- User analytics

---

## 11. Future Enhancements

### 11.1 Feature Roadmap

**Short-term:**
- Additional ML models (SVM, Neural Networks)
- Cross-validation support
- Hyperparameter tuning interface
- Model explainability (LIME/SHAP)

**Medium-term:**
- Multi-class sentiment (positive/neutral/negative)
- Aspect-based sentiment analysis
- Real-time prediction API
- Model versioning system

**Long-term:**
- Deep learning models (BERT, Transformers)
- Multi-language support
- Automated retraining pipeline
- A/B testing framework

### 11.2 Technical Debt

**Current Limitations:**
- No automated testing suite
- Limited error recovery
- Manual model deployment
- No model versioning
- Fixed text preprocessing pipeline

**Improvement Areas:**
- Add comprehensive unit tests
- Implement CI/CD pipeline
- Add configuration management
- Refactor for better extensibility
- Add performance benchmarking

---

## 12. Assignment Component Checklist

### Component 1: Problem Domain Description
- [x] Business context explained
- [x] Dataset details provided
- [x] Target variable defined
- [x] Challenges identified

### Component 2: User Interface Layout/Design
- [x] 5-page navigation structure
- [x] Detailed wireframes per page
- [x] Component specifications
- [x] Design system documented
- [x] User flows defined

### Component 3: Data Cleaning Techniques
- [x] Field detection system
- [x] Rating normalization
- [x] Null handling
- [x] Class balancing
- [x] Text preprocessing pipeline

### Component 4: Visualization Techniques
- [x] Data exploration plots
- [x] Model performance visualizations
- [x] Interactive dashboards
- [x] Export functionality

### Component 5: Data Mining Algorithms
- [x] Naive Bayes implementation
- [x] Logistic Regression implementation
- [x] Random Forest implementation
- [x] Evaluation metrics
- [x] Comparison framework

### Component 6: Actual Implementation
- [x] Training module (main.py)
- [x] Deployment module (app.py)
- [x] Modular codebase
- [x] End-to-end workflow

---

## Document Version

**Version:** 1.0
**Date:** 2025-10-26
**Author:** Development Team
**Status:** Complete

---

## References

### Technical Documentation
- Scikit-learn documentation: https://scikit-learn.org/
- Streamlit documentation: https://docs.streamlit.io/
- Pandas documentation: https://pandas.pydata.org/

### Academic References
- TF-IDF vectorization: Ramos, J. (2003). Using TF-IDF to determine word relevance
- Naive Bayes for text: McCallum, A., & Nigam, K. (1998). A comparison of event models
- Logistic Regression: Hosmer, D. W., & Lemeshow, S. (2000). Applied Logistic Regression
- Random Forests: Breiman, L. (2001). Random Forests. Machine Learning

### Dataset References
- Amazon Reviews dataset: https://www.kaggle.com/datasets
- Multi-domain sentiment datasets: Various Kaggle and UCI ML Repository sources
