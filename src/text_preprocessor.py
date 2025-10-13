import re
from typing import List

class TextPreprocessor:
    """
    Preprocesses text for sentiment analysis.
    - Removes HTML tags
    - Converts to lowercase
    - Removes punctuation (keeps apostrophes)
    - Removes stop words
    - Filters short words
    """
    
    def __init__(self, use_lemmatization: bool = False):
        # Regex patterns
        self.html_pattern = re.compile(r'<[^>]+>')
        self.punct_pattern = re.compile(r"[^a-zA-Z0-9\s']")
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Minimal stop words (function words + domain-neutral terms)
        self.stop_words = {
            # Basic function words
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'these', 'them', 'had', 'can', 'get', 'than', 'only', 
            'also', 'other', 'time', 'after', 'because', 'even', 'really', 'amazon', 'bought', 'buying', 
            'tried', 'made', 'found', 'ordered', 'first', 'used',
            
            # Product-review neutral terms
            'product', 'item', 'purchased', 'ordered', 'received',
            
            # Pronouns (non-sentiment)
            'i', 'you', 'his', 'her', 'she', 'they', 'this',
            
            # Additional common words
            'but', 'not', 'one', 'have', 'all', 'who', 'like', 'out',
            'just', 'there', "it's", 'about', 'when', 'what', 'more',
            'very', 'some', 'would', 'been', 'which', 'were', 'their'
        }
        
        # Optional lemmatization (can be added later with nltk)
        self.use_lemmatization = use_lemmatization
        if use_lemmatization:
            try:
                from nltk.stem import WordNetLemmatizer
                self.lemmatizer = WordNetLemmatizer()
            except ImportError:
                print("Warning: nltk not available, lemmatization disabled")
                self.use_lemmatization = False
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text and return cleaned string.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string with words separated by spaces
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        cleaned = self.html_pattern.sub(' ', text)
        
        # Convert to lowercase
        cleaned = cleaned.lower()
        
        # Remove punctuation (keep apostrophes for contractions)
        cleaned = self.punct_pattern.sub(' ', cleaned)
        
        # Normalize whitespace
        cleaned = self.whitespace_pattern.sub(' ', cleaned).strip()
        
        # Tokenize
        words = cleaned.split()
        
        # Filter: remove short words and stop words
        filtered = [
            word for word in words
            if len(word) > 2 and word not in self.stop_words
        ]
        
        # Optional lemmatization
        if self.use_lemmatization and hasattr(self, 'lemmatizer'):
            filtered = [self.lemmatizer.lemmatize(word) for word in filtered]
        
        return ' '.join(filtered)
    
    def extract_features(self, text: str) -> List[str]:
        """
        Extract word features from text.
        
        Args:
            text: Raw text string
            
        Returns:
            List of preprocessed words
        """
        preprocessed = self.preprocess(text)
        return preprocessed.split() if preprocessed else []


# Test the preprocessor
if __name__ == "__main__":
    processor = TextPreprocessor()
    
    # Test with product review
    sample = """
    I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates  this product better than most.
    """
    
    print("Original text:")
    print(sample)
    print("\nPreprocessed text:")
    print(processor.preprocess(sample))
    print("\nExtracted features:")
    print(processor.extract_features(sample))