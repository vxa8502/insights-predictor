import pandas as pd
import logging
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ReviewData:
    text: str
    sentiment: str

class AmazonDataLoader:
    """
    Loads and processes Amazon product reviews dataset.
    Converts 1-5 star ratings to binary sentiment (skips 3-star reviews).
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)
        self.stats = {
            'total_records': 0,
            'successful_records': 0,
            'skipped_neutral': 0,
            'null_or_empty_text': 0,
            'invalid_score': 0,
            'positive_count': 0,
            'negative_count': 0
        }
        
    def load_data(self, sample_size: int = None, balance: bool = True) -> Tuple[List[ReviewData], dict]:
        """
        Load Amazon reviews from CSV file.
        
        Args:
            sample_size: Optional - randomly sample this many records before processing
            balance: If True, balance positive/negative classes
            
        Returns:
            Tuple of (list of ReviewData objects, statistics dict)
        """
        self.logger.info(f"Loading data from {self.file_path}")
        
        # Read CSV
        df = pd.read_csv(
            self.file_path,
            encoding='utf-8',
            on_bad_lines='skip',
            low_memory=False
        )
        
        self.logger.info(f"Loaded {len(df)} total records")
        self.stats['total_records'] = len(df)
        
        # Verify required columns
        if 'Text' not in df.columns or 'Score' not in df.columns:
            raise ValueError(f"Required columns 'Text' and 'Score' not found. Available: {list(df.columns)}")
        
        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            self.logger.info(f"Sampled {sample_size} records")
        
        # Convert Score to sentiment
        df['sentiment'] = df['Score'].apply(self._score_to_sentiment)
        
        # Track 3-star reviews
        self.stats['skipped_neutral'] = (df['Score'] == 3).sum()
        
        # Remove null sentiments (3-star reviews)
        df = df[df['sentiment'].notna()].copy()
        
        # Remove null/empty text
        initial_count = len(df)
        df = df[df['Text'].notna()].copy()
        df = df[df['Text'].str.strip() != ''].copy()
        self.stats['null_or_empty_text'] = initial_count - len(df)
        
        # Balance classes if requested
        if balance:
            df = self._balance_classes(df)
        
        # Count final distribution
        self.stats['positive_count'] = (df['sentiment'] == 'positive').sum()
        self.stats['negative_count'] = (df['sentiment'] == 'negative').sum()
        
        # Convert to ReviewData objects
        reviews = [
            ReviewData(text=str(row['Text']), sentiment=row['sentiment']) 
            for _, row in df.iterrows()
        ]
        
        self.stats['successful_records'] = len(reviews)
        self._log_stats()
        
        return reviews, self.stats
    
    def _score_to_sentiment(self, score) -> str:
        """
        Convert Amazon 1-5 star rating to sentiment.
        1-2 stars = negative
        3 stars = None (skip)
        4-5 stars = positive
        """
        try:
            score = float(score)
            if score <= 2.0:
                return 'negative'
            elif score >= 4.0:
                return 'positive'
            else:
                return None  # Skip 3-star
        except (ValueError, TypeError):
            self.stats['invalid_score'] += 1
            return None
    
    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance positive and negative classes to 50/50 distribution."""
        pos = df[df['sentiment'] == 'positive']
        neg = df[df['sentiment'] == 'negative']
        
        self.logger.info(f"Before balancing - Positive: {len(pos)}, Negative: {len(neg)}")
        
        # Find minimum class size
        min_size = min(len(pos), len(neg))
        
        # Cap at 50k per class for performance
        max_per_class = min(min_size, 50000)
        
        # Sample equally from both classes
        balanced = pd.concat([
            pos.sample(n=max_per_class, random_state=42),
            neg.sample(n=max_per_class, random_state=42)
        ])
        
        # Shuffle
        balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        self.logger.info(f"After balancing - Positive: {max_per_class}, Negative: {max_per_class}")
        
        return balanced
    
    def _log_stats(self):
        """Log loading statistics summary."""
        self.logger.info("DATA LOADING SUMMARY")
        self.logger.info(f"Total records attempted: {self.stats['total_records']}")
        self.logger.info(f"Successfully loaded: {self.stats['successful_records']}")
        
        if self.stats['total_records'] > 0:
            success_rate = self.stats['successful_records'] / self.stats['total_records'] * 100
            self.logger.info(f"Success rate: {success_rate:.1f}%")
        
        self.logger.info(f"Skipped 3-star reviews: {self.stats['skipped_neutral']}")
        self.logger.info(f"Null/empty text: {self.stats['null_or_empty_text']}")
        self.logger.info(f"Invalid scores: {self.stats['invalid_score']}")
        self.logger.info(f"Final distribution - Positive: {self.stats['positive_count']}, Negative: {self.stats['negative_count']}")
        self.logger.info("=" * 60)


# Test the data loader
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        loader = AmazonDataLoader('data/Reviews.csv')
        reviews, stats = loader.load_data(sample_size=10000)
        
        print(f"\nLoaded {len(reviews)} reviews")
        print(f"\nFirst 3 reviews:")
        for i, review in enumerate(reviews[:3], 1):
            print(f"\n{i}. Sentiment: {review.sentiment}")
            print(f"   Text: {review.text[:100]}...")
            
    except Exception as e:
        print(f"Error: {e}")