import pandas as pd
import logging
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .field_extractor import extract_text_field, extract_sentiment_field, get_dataset_info

@dataclass
class ReviewData:
    text: str
    sentiment: str

class DataLoader:
    """
    Flexible data loader that handles multiple dataset formats.
    Uses smart field extraction to automatically detect text and sentiment columns.
    Supports numeric ratings (1-5 stars) and text labels (positive/negative).
    """

    def __init__(
        self,
        file_path: str,
        additional_text_fields: Tuple[str, ...] = (),
        additional_sentiment_fields: Tuple[str, ...] = ()
    ):
        """
        Initialize the data loader.

        Args:
            file_path: Path to the CSV file
            additional_text_fields: Extra field names to try for text column
            additional_sentiment_fields: Extra field names to try for sentiment column
        """
        self.file_path = file_path
        self.additional_text_fields = additional_text_fields
        self.additional_sentiment_fields = additional_sentiment_fields
        self.logger = logging.getLogger(__name__)

        self.text_column = None
        self.sentiment_column = None
        self.dataset_type = None

        self.stats = {
            'total_records': 0,
            'successful_records': 0,
            'skipped_neutral': 0,
            'null_or_empty_text': 0,
            'invalid_score': 0,
            'positive_count': 0,
            'negative_count': 0,
            'field_extraction': {}
        }

    def load_data(self, sample_size: int = None, balance: bool = True) -> Tuple[List[ReviewData], dict]:
        """
        Load reviews from CSV file with flexible field mapping.

        Args:
            sample_size: Optional - randomly sample this many records before processing
            balance: If True, balance positive/negative classes

        Returns:
            Tuple of (list of ReviewData objects, statistics dict)
        """
        self.logger.info(f"Loading data from {self.file_path}")

        # Read CSV with multiple encoding attempts for robustness
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(
                    self.file_path,
                    encoding=encoding,
                    on_bad_lines='skip',
                    low_memory=False
                )
                if encoding != 'utf-8':
                    self.logger.info(f"Loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError(f"Could not read file with any encoding: {encodings}")

        self.logger.info(f"Loaded {len(df)} total records")
        self.stats['total_records'] = len(df)

        # Use flexible field extraction to find text and sentiment columns
        self.text_column, text_result = extract_text_field(
            df,
            additional_names=self.additional_text_fields,
            fallback_position=0
        )

        self.sentiment_column, sentiment_result = extract_sentiment_field(
            df,
            additional_names=self.additional_sentiment_fields,
            fallback_position=1
        )

        # Get dataset info
        dataset_info = get_dataset_info(df)
        self.dataset_type = dataset_info['dataset_type']

        # Log what we found
        self.logger.info(f"Detected dataset type: {self.dataset_type}")
        self.logger.info(f"Text column: '{self.text_column}' (found by {text_result.found_by})")
        self.logger.info(f"Sentiment column: '{self.sentiment_column}' (found by {sentiment_result.found_by})")

        # Store field extraction metadata
        self.stats['field_extraction'] = {
            'text_column': self.text_column,
            'text_found_by': text_result.found_by,
            'sentiment_column': self.sentiment_column,
            'sentiment_found_by': sentiment_result.found_by,
            'dataset_type': self.dataset_type
        }

        # VALIDATION: Ensure dataset is suitable for text sentiment analysis
        self._validate_text_field(df)
        self._validate_sentiment_field(df)

        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            self.logger.info(f"Sampled {sample_size} records")

        # Normalize sentiment column - convert to standard format
        df['sentiment_normalized'] = df[self.sentiment_column].apply(self._normalize_sentiment)

        # Track neutral/skipped reviews
        neutral_count_before = df['sentiment_normalized'].isna().sum()
        self.stats['skipped_neutral'] = neutral_count_before

        # Remove null sentiments (neutral reviews or invalid values)
        df = df[df['sentiment_normalized'].notna()].copy()

        # Remove null/empty text
        initial_count = len(df)
        df = df[df[self.text_column].notna()].copy()
        df = df[df[self.text_column].astype(str).str.strip() != ''].copy()
        self.stats['null_or_empty_text'] = initial_count - len(df)

        # Balance classes if requested
        if balance:
            df = self._balance_classes(df)

        # Count final distribution
        self.stats['positive_count'] = (df['sentiment_normalized'] == 'positive').sum()
        self.stats['negative_count'] = (df['sentiment_normalized'] == 'negative').sum()

        # Convert to ReviewData objects
        reviews = [
            ReviewData(text=str(row[self.text_column]), sentiment=row['sentiment_normalized'])
            for _, row in df.iterrows()
        ]

        self.stats['successful_records'] = len(reviews)

        # VALIDATION: Ensure we have enough valid records
        self._validate_final_dataset(reviews)

        self._log_stats()

        return reviews, self.stats

    def _validate_text_field(self, df: pd.DataFrame) -> None:
        """
        Validate that the detected text field contains actual text content.

        Raises:
            ValueError: If text field appears to be numeric IDs or non-text data
        """
        sample_size = min(100, len(df))
        sample = df[self.text_column].dropna().head(sample_size)

        if len(sample) == 0:
            raise ValueError(
                f"Dataset validation failed!\n\n"
                f"Problem: The text field '{self.text_column}' is completely empty.\n\n"
                f"Required: A column with review text (sentences, paragraphs, comments)\n\n"
                f"Available columns: {', '.join(df.columns)}\n\n"
                f"Suggestion: Check your CSV file has a column with actual review text."
            )

        # Check if column contains mostly numeric values (like IDs)
        numeric_count = 0
        for val in sample:
            try:
                # Try converting to float - if successful, it's numeric
                float(str(val).strip())
                numeric_count += 1
            except (ValueError, AttributeError):
                pass

        numeric_ratio = numeric_count / len(sample)

        if numeric_ratio > 0.9:
            raise ValueError(
                f"Dataset validation failed!\n\n"
                f"Problem: The detected text field '{self.text_column}' appears to contain "
                f"numeric IDs or numbers, not review text.\n"
                f"  Sample values: {list(sample.head(3))}\n\n"
                f"Required: A text column with review content (strings, sentences, paragraphs)\n\n"
                f"Available columns: {', '.join(df.columns)}\n\n"
                f"Common text column names: review, text, comment, review_text, content, message\n\n"
                f"Suggestion: Ensure your dataset has actual review text, not just IDs or ratings."
            )

        # Check average text length - very short text (< 5 chars avg) is suspicious
        avg_length = sample.astype(str).str.len().mean()

        if avg_length < 5:
            raise ValueError(
                f"Dataset validation failed!\n\n"
                f"Problem: The text field '{self.text_column}' has very short content "
                f"(average {avg_length:.1f} characters).\n"
                f"  Sample values: {list(sample.head(3))}\n\n"
                f"Required: Meaningful review text (typically 20+ characters)\n\n"
                f"Available columns: {', '.join(df.columns)}\n\n"
                f"Suggestion: This column may contain codes or labels instead of review text."
            )

        self.logger.info(f"Text field validation passed (avg length: {avg_length:.0f} chars)")

    def _validate_sentiment_field(self, df: pd.DataFrame) -> None:
        """
        Validate that we can extract valid sentiments from the detected sentiment field.

        Raises:
            ValueError: If sentiment field cannot be parsed or has no valid values
        """
        sample_size = min(1000, len(df))
        sample = df[self.sentiment_column].dropna().head(sample_size)

        if len(sample) == 0:
            raise ValueError(
                f"Dataset validation failed!\n\n"
                f"Problem: The sentiment field '{self.sentiment_column}' is completely empty.\n\n"
                f"Required: A column with ratings or sentiment labels\n\n"
                f"Available columns: {', '.join(df.columns)}\n\n"
                f"Suggestion: Check your CSV file has a rating or sentiment column."
            )

        # Try to normalize a sample and see how many succeed
        valid_count = 0
        sample_values = []

        for val in sample:
            normalized = self._normalize_sentiment(val)
            if normalized is not None:
                valid_count += 1
            if len(sample_values) < 5:
                sample_values.append(val)

        valid_ratio = valid_count / len(sample)

        if valid_ratio < 0.1:
            raise ValueError(
                f"Dataset validation failed!\n\n"
                f"Problem: Cannot extract valid sentiments from '{self.sentiment_column}'. "
                f"Only {valid_ratio:.1%} of values are recognized.\n"
                f"  Sample values: {sample_values}\n\n"
                f"Supported formats:\n"
                f"  - Numeric ratings: 1-5 (stars), 1-10 (scale), 0-100 (percentage)\n"
                f"  - Star text: '5 stars', '1 star'\n"
                f"  - Text labels: 'positive', 'negative', 'pos', 'neg'\n"
                f"  - Binary: 0/1, True/False\n\n"
                f"Available columns: {', '.join(df.columns)}\n\n"
                f"Suggestion: Check if your dataset uses a different rating column or format."
            )

        self.logger.info(f"Sentiment field validation passed ({valid_ratio:.1%} valid values)")

    def _validate_final_dataset(self, reviews: List[ReviewData]) -> None:
        """
        Validate that we have enough records after all processing.

        Raises:
            ValueError: If final dataset is too small or empty
        """
        if len(reviews) == 0:
            raise ValueError(
                f"Dataset validation failed!\n\n"
                f"Problem: No valid reviews remain after processing.\n\n"
                f"Possible causes:\n"
                f"  - All ratings are neutral (e.g., all 3-star reviews)\n"
                f"  - Rating format not recognized\n"
                f"  - Text field contains empty or invalid content\n\n"
                f"Processing stats:\n"
                f"  - Total records: {self.stats['total_records']}\n"
                f"  - Skipped neutral: {self.stats['skipped_neutral']}\n"
                f"  - Null/empty text: {self.stats['null_or_empty_text']}\n"
                f"  - Invalid scores: {self.stats['invalid_score']}\n\n"
                f"Suggestion: Review your dataset to ensure it has valid ratings and text content."
            )

        if len(reviews) < 100:
            self.logger.warning(
                f"Warning: Only {len(reviews)} valid reviews found. "
                f"This may be too small for reliable model training. "
                f"Consider using a larger dataset (recommended: 1000+ reviews)."
            )

    def _normalize_sentiment(self, value) -> Optional[str]:
        """
        Normalize sentiment/rating value to 'positive' or 'negative'.

        Handles:
        - Numeric ratings (1-5 stars): 1-2=negative, 3=skip, 4-5=positive
        - 1-10 rating scale: 1-4=negative, 5-6=skip, 7-10=positive
        - Star text format (e.g., "5 stars", "1 star"): same as 1-5 ratings
        - 0-100 scale (e.g. wine points): 70-100=positive, 50-69=skip, 0-49=negative
        - Text labels: 'positive', 'negative', 'pos', 'neg', etc.
        - Binary values: 0/1, True/False

        Returns:
            'positive', 'negative', or None (for neutral/invalid)
        """
        if pd.isna(value):
            return None

        # Try numeric conversion first
        try:
            numeric_value = float(value)

            # Handle 1-5 star ratings
            if 1.0 <= numeric_value <= 5.0:
                if numeric_value <= 2.0:
                    return 'negative'
                elif numeric_value >= 4.0:
                    return 'positive'
                else:
                    return None  # Skip 3-star (neutral)

            # Handle 1-10 rating scale (airline reviews, hotel ratings, etc.)
            elif 1.0 <= numeric_value <= 10.0:
                if numeric_value <= 4.0:
                    return 'negative'
                elif numeric_value >= 7.0:
                    return 'positive'
                else:
                    return None  # Skip 5-6 (neutral)

            # Handle 0-100 scale (wine points, review scores, game ratings, etc.)
            elif 0.0 <= numeric_value <= 100.0:
                if numeric_value >= 70.0:
                    return 'positive'
                elif numeric_value >= 50.0:
                    return None  # Skip neutral range (50-69)
                else:
                    return 'negative'

            # Unknown numeric scale (values > 100 or negative values)
            else:
                self.stats['invalid_score'] += 1
                return None

        except (ValueError, TypeError):
            # Not numeric - treat as text label
            pass

        # Handle text labels
        value_str = str(value).lower().strip()

        # Handle "X star" or "X stars" format (e.g., "5 stars", "1 star")
        if 'star' in value_str:
            star_match = re.search(r'(\d+(?:\.\d+)?)\s*stars?', value_str)
            if star_match:
                try:
                    star_value = float(star_match.group(1))
                    # Apply same 1-5 star logic
                    if 1.0 <= star_value <= 5.0:
                        if star_value <= 2.0:
                            return 'negative'
                        elif star_value >= 4.0:
                            return 'positive'
                        else:
                            return None  # Skip 3-star (neutral)
                except (ValueError, IndexError):
                    pass

        # Positive indicators
        if value_str in ['positive', 'pos', '1', 'true', 'good', 'like', 'thumbs up']:
            return 'positive'

        # Negative indicators
        if value_str in ['negative', 'neg', '0', 'false', 'bad', 'dislike', 'thumbs down']:
            return 'negative'

        # Neutral or unknown
        if value_str in ['neutral', 'mixed', 'unknown', '']:
            return None

        # Couldn't parse
        self.stats['invalid_score'] += 1
        return None

    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance positive and negative classes to 50/50 distribution."""
        pos = df[df['sentiment_normalized'] == 'positive']
        neg = df[df['sentiment_normalized'] == 'negative']

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
    logger = logging.getLogger(__name__)

    try:
        loader = DataLoader('data/Reviews.csv')
        reviews, stats = loader.load_data(sample_size=10000)

        logger.info(f"\nLoaded {len(reviews)} reviews")
        logger.info(f"\nFirst 3 reviews:")
        for i, review in enumerate(reviews[:3], 1):
            logger.info(f"\n{i}. Sentiment: {review.sentiment}")
            logger.info(f"   Text: {review.text[:100]}...")

    except Exception as e:
        logger.error(f"Error: {e}")
