"""
Flexible field extraction system for handling multiple dataset formats.
Inspired by the FieldExtractor pattern - tries multiple field names with fallback to positional index.
"""

from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
import logging


@dataclass
class FieldMapping:
    """Container for field name variants that can be extended with additional names."""

    names: Tuple[str, ...]

    def __init__(self, *names: str):
        """Initialize with a list of field name variants."""
        self.names = tuple(name.lower() for name in names)

    def with_additional(self, *additional_names: str) -> 'FieldMapping':
        """Return new FieldMapping with additional field names appended."""
        all_names = self.names + tuple(name.lower() for name in additional_names)
        new_mapping = FieldMapping()
        new_mapping.names = all_names
        return new_mapping

    def __iter__(self):
        """Allow iteration over field names."""
        return iter(self.names)


@dataclass
class ExtractionResult:
    """Result of field extraction containing the value and metadata about how it was found."""

    value: Any
    found_by: str  # 'name' or 'position'
    field_name: Optional[str] = None  # Actual field name used if found by name
    position: Optional[int] = None  # Position used if found by position


class FieldExtractor:
    """
    Smart field extractor that handles multiple naming conventions.

    Extraction Strategy:
    1. Try to find field by name (case-insensitive) from list of known variants
    2. Fall back to positional index if name-based lookup fails
    3. Return metadata about how the field was found
    """

    # Pre-defined text field mappings - handles many different naming conventions
    # Note: "description" removed - too generic, often refers to product descriptions not reviews
    TEXT_FIELDS = FieldMapping(
        "text",
        "content",
        "review_text",
        "reviewtext",
        "review",
        "reviewbody",
        "review body",
        "review text",  # Space variant
        "comment",
        "tweet_text",
        "tweet",
        "message",
        "post",
        "body",
        "summary",
        "reviews.text",
        "review_comment",
        "customer_review",
        "review/text",  # Slash notation
        "review_text"
    )

    # Pre-defined sentiment/label field mappings - handles many different conventions
    SENTIMENT_FIELDS = FieldMapping(
        "sentiment",
        "label",
        "polarity",
        "class",
        "rating",
        "score",
        "points",
        "emotion",
        "grade",
        "overall",
        "reviews.rating",
        "overall_rating",
        "overallrating",
        "recommended",
        "stars",
        "review/score",  # Slash notation
        "review_score"
    )

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_field(
        self,
        df: pd.DataFrame,
        field_mapping: FieldMapping,
        fallback_position: Optional[int] = None,
        required: bool = True
    ) -> ExtractionResult:
        """
        Extract a field from DataFrame using flexible field mapping.

        Args:
            df: DataFrame to extract from
            field_mapping: FieldMapping with possible field names to try
            fallback_position: Column index to use if no field names match
            required: If True, raises error when field not found. If False, returns None.

        Returns:
            ExtractionResult with the column name and how it was found

        Raises:
            ValueError: If required=True and field cannot be found
        """
        # Convert all column names to lowercase for case-insensitive matching
        columns_lower = {col.lower(): col for col in df.columns}

        # Try each field name in the mapping
        for field_name in field_mapping:
            if field_name in columns_lower:
                actual_col_name = columns_lower[field_name]
                self.logger.debug(f"Found field by name: '{actual_col_name}' (matched '{field_name}')")
                return ExtractionResult(
                    value=actual_col_name,
                    found_by='name',
                    field_name=actual_col_name
                )

        # Try fallback position if provided
        if fallback_position is not None:
            if 0 <= fallback_position < len(df.columns):
                col_name = df.columns[fallback_position]
                self.logger.debug(f"Found field by position: '{col_name}' at index {fallback_position}")
                return ExtractionResult(
                    value=col_name,
                    found_by='position',
                    position=fallback_position
                )

        # Field not found
        if required:
            attempted_names = ", ".join(field_mapping.names)
            available_cols = ", ".join(df.columns)
            raise ValueError(
                f"Could not find field. Tried names: [{attempted_names}]. "
                f"Available columns: [{available_cols}]"
            )

        return ExtractionResult(value=None, found_by='not_found')

    def extract_string(
        self,
        df: pd.DataFrame,
        field_mapping: FieldMapping,
        fallback_position: Optional[int] = None,
        required: bool = True
    ) -> Tuple[str, ExtractionResult]:
        """
        Extract a string field and return both the column name and extraction metadata.

        Args:
            df: DataFrame to extract from
            field_mapping: FieldMapping with possible field names
            fallback_position: Column index fallback
            required: Whether field is required

        Returns:
            Tuple of (column_name, ExtractionResult)
        """
        result = self.extract_field(df, field_mapping, fallback_position, required)
        return result.value, result

    def detect_dataset_type(self, df: pd.DataFrame) -> str:
        """
        Auto-detect dataset type based on column names.

        Returns:
            String indicating dataset type: 'twitter', 'product_reviews', 'movie_reviews', 'generic'
        """
        columns_lower = {col.lower() for col in df.columns}

        # Twitter indicators
        twitter_indicators = {'tweet', 'tweet_text', 'user', 'retweet', 'hashtag'}
        if columns_lower & twitter_indicators:
            return 'twitter'

        # Product review indicators
        product_indicators = {'stars', 'star_rating', 'product', 'verified_purchase'}
        if columns_lower & product_indicators:
            return 'product_reviews'

        # Movie review indicators
        movie_indicators = {'movie_review', 'film_review', 'movie', 'film', 'imdb'}
        if columns_lower & movie_indicators:
            return 'movie_reviews'

        return 'generic'

    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive information about the dataset structure.

        Returns:
            Dictionary with dataset metadata
        """
        text_col, text_result = self.extract_string(df, self.TEXT_FIELDS, fallback_position=0, required=False)
        sentiment_col, sentiment_result = self.extract_string(df, self.SENTIMENT_FIELDS, fallback_position=1, required=False)

        return {
            'dataset_type': self.detect_dataset_type(df),
            'total_columns': len(df.columns),
            'column_names': list(df.columns),
            'text_field': {
                'column': text_col,
                'found_by': text_result.found_by,
                'matched_name': text_result.field_name
            },
            'sentiment_field': {
                'column': sentiment_col,
                'found_by': sentiment_result.found_by,
                'matched_name': sentiment_result.field_name
            }
        }


# Convenience singleton instance
_extractor = FieldExtractor()


def extract_text_field(
    df: pd.DataFrame,
    additional_names: Tuple[str, ...] = (),
    fallback_position: int = 0
) -> Tuple[str, ExtractionResult]:
    """
    Convenience function to extract text field with optional additional names.

    Args:
        df: DataFrame to extract from
        additional_names: Additional field names to try beyond defaults
        fallback_position: Column index fallback (default: 0)

    Returns:
        Tuple of (column_name, ExtractionResult)
    """
    field_mapping = FieldExtractor.TEXT_FIELDS
    if additional_names:
        field_mapping = field_mapping.with_additional(*additional_names)

    return _extractor.extract_string(df, field_mapping, fallback_position)


def extract_sentiment_field(
    df: pd.DataFrame,
    additional_names: Tuple[str, ...] = (),
    fallback_position: int = 1
) -> Tuple[str, ExtractionResult]:
    """
    Convenience function to extract sentiment/label field with optional additional names.

    Args:
        df: DataFrame to extract from
        additional_names: Additional field names to try beyond defaults
        fallback_position: Column index fallback (default: 1)

    Returns:
        Tuple of (column_name, ExtractionResult)
    """
    field_mapping = FieldExtractor.SENTIMENT_FIELDS
    if additional_names:
        field_mapping = field_mapping.with_additional(*additional_names)

    return _extractor.extract_string(df, field_mapping, fallback_position)


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Convenience function to get dataset information.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with dataset metadata
    """
    return _extractor.get_dataset_info(df)
