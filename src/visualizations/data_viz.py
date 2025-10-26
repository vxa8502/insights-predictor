"""
Data Visualization Module for Sentiment Analysis
Provides functions for exploring and visualizing review data
"""

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from typing import List, Tuple
from collections import Counter
import logging


class DataVisualizer:
    """
    Handles all data exploration and visualization tasks.
    Creates plots for sentiment distribution, text analysis, and feature exploration.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer with plotting style.
        
        Args:
            style: Matplotlib style to use for plots
        """
        self.logger = logging.getLogger(__name__)
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Color scheme for consistency
        self.colors = {
            'positive': '#2ecc71',  # Green
            'negative': '#e74c3c',  # Red
            'neutral': '#95a5a6'    # Gray
        }
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, 
                                   save_path: str = None) -> plt.Figure:
        """
        Create bar chart showing distribution of sentiment classes.
        
        Args:
            df: DataFrame with 'sentiment' column
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Count sentiments
        sentiment_counts = df['sentiment'].value_counts()
        
        # Create bar plot
        bars = ax.bar(sentiment_counts.index, sentiment_counts.values,
                     color=[self.colors[sent] for sent in sentiment_counts.index],
                     alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add percentage labels
        total = sentiment_counts.sum()
        for i, (sentiment, count) in enumerate(sentiment_counts.items()):
            percentage = (count / total) * 100
            ax.text(i, count * 0.5, f'{percentage:.1f}%',
                   ha='center', va='center', fontsize=14, 
                   color='white', fontweight='bold')
        
        ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
        ax.set_title('Sentiment Distribution in Dataset', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Sentiment distribution plot saved to {save_path}")
        
        return fig
    
    def generate_wordclouds(self, df: pd.DataFrame, text_column: str = 'Text',
                           save_path_prefix: str = None) -> Tuple[plt.Figure, plt.Figure]:
        """
        Generate separate word clouds for positive and negative reviews.
        
        Args:
            df: DataFrame with text and sentiment columns
            text_column: Name of column containing review text
            save_path_prefix: Optional prefix for saving figures (e.g., 'plots/wordcloud')
            
        Returns:
            Tuple of (positive wordcloud figure, negative wordcloud figure)
        """
        # Separate positive and negative reviews
        positive_text = ' '.join(df[df['sentiment'] == 'positive'][text_column].astype(str))
        negative_text = ' '.join(df[df['sentiment'] == 'negative'][text_column].astype(str))
        
        # Create word clouds
        wordcloud_config = {
            'width': 800,
            'height': 400,
            'background_color': 'white',
            'max_words': 100,
            'relative_scaling': 0.5,
            'min_font_size': 10
        }
        
        # Positive word cloud
        fig_pos, ax_pos = plt.subplots(figsize=(12, 6))
        wc_positive = WordCloud(**wordcloud_config, colormap='Greens').generate(positive_text)
        ax_pos.imshow(wc_positive, interpolation='bilinear')
        ax_pos.axis('off')
        ax_pos.set_title('Most Frequent Words in Positive Reviews', 
                        fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path_prefix:
            fig_pos.savefig(f"{save_path_prefix}_positive.png", dpi=300, bbox_inches='tight')
            self.logger.info(f"Positive wordcloud saved to {save_path_prefix}_positive.png")
        
        # Negative word cloud
        fig_neg, ax_neg = plt.subplots(figsize=(12, 6))
        wc_negative = WordCloud(**wordcloud_config, colormap='Reds').generate(negative_text)
        ax_neg.imshow(wc_negative, interpolation='bilinear')
        ax_neg.axis('off')
        ax_neg.set_title('Most Frequent Words in Negative Reviews', 
                        fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path_prefix:
            fig_neg.savefig(f"{save_path_prefix}_negative.png", dpi=300, bbox_inches='tight')
            self.logger.info(f"Negative wordcloud saved to {save_path_prefix}_negative.png")
        
        return fig_pos, fig_neg
    
    def plot_review_length_distribution(self, df: pd.DataFrame, 
                                       text_column: str = 'Text',
                                       save_path: str = None) -> plt.Figure:
        """
        Create box plot showing distribution of review lengths by sentiment.
        
        Args:
            df: DataFrame with text and sentiment columns
            text_column: Name of column containing review text
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Calculate text lengths
        df = df.copy()
        df['text_length'] = df[text_column].astype(str).apply(len)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot
        sns.boxplot(data=df, x='sentiment', y='text_length', ax=ax1,
                   palette=[self.colors['negative'], self.colors['positive']])
        ax1.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Review Length (characters)', fontsize=12, fontweight='bold')
        ax1.set_title('Review Length Distribution by Sentiment', 
                     fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Histogram with overlapping distributions
        positive_lengths = df[df['sentiment'] == 'positive']['text_length']
        negative_lengths = df[df['sentiment'] == 'negative']['text_length']
        
        ax2.hist(positive_lengths, bins=50, alpha=0.6, 
                label='Positive', color=self.colors['positive'], edgecolor='black')
        ax2.hist(negative_lengths, bins=50, alpha=0.6, 
                label='Negative', color=self.colors['negative'], edgecolor='black')
        
        ax2.set_xlabel('Review Length (characters)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Overlapping Length Distributions', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Review length distribution plot saved to {save_path}")
        
        return fig
    
    def plot_top_words(self, df: pd.DataFrame, text_column: str = 'Text',
                      top_n: int = 20, save_path: str = None) -> plt.Figure:
        """
        Create bar chart showing most frequent words by sentiment.
        
        Args:
            df: DataFrame with text and sentiment columns
            text_column: Name of column containing preprocessed text
            top_n: Number of top words to display
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get word frequencies for each sentiment
        for ax, sentiment, color in [(ax1, 'positive', self.colors['positive']),
                                      (ax2, 'negative', self.colors['negative'])]:
            # Collect all words for this sentiment
            all_words = []
            for text in df[df['sentiment'] == sentiment][text_column].astype(str):
                all_words.extend(text.split())
            
            # Count frequencies
            word_freq = Counter(all_words)
            top_words = word_freq.most_common(top_n)
            
            words = [word for word, _ in top_words]
            counts = [count for _, count in top_words]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(words))
            ax.barh(y_pos, counts, color=color, alpha=0.8, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'Top {top_n} Words in {sentiment.capitalize()} Reviews',
                        fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, count in enumerate(counts):
                ax.text(count, i, f'  {count:,}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Top words plot saved to {save_path}")
        
        return fig
    
    def plot_rating_distribution(self, df: pd.DataFrame, 
                                 rating_column: str = 'Score',
                                 save_path: str = None) -> plt.Figure:
        """
        Create histogram showing distribution of star ratings.
        
        Args:
            df: DataFrame with rating column
            rating_column: Name of column containing ratings (1-5)
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Count ratings
        rating_counts = df[rating_column].value_counts().sort_index()
        
        # Color bars by sentiment (1-2: red, 3: gray, 4-5: green)
        colors = []
        for rating in rating_counts.index:
            if rating <= 2:
                colors.append(self.colors['negative'])
            elif rating >= 4:
                colors.append(self.colors['positive'])
            else:
                colors.append(self.colors['neutral'])
        
        # Create bar plot
        bars = ax.bar(rating_counts.index, rating_counts.values,
                     color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Star Rating', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Star Ratings', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Rating distribution plot saved to {save_path}")
        
        return fig
    
    def create_exploratory_dashboard(self, df: pd.DataFrame, 
                                     text_column: str = 'Text',
                                     save_path: str = None):
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            df: DataFrame with review data
            text_column: Name of column containing review text
            save_path: Optional path to save figure
        """
        # Calculate text lengths if not present
        if 'text_length' not in df.columns:
            df = df.copy()
            df['text_length'] = df[text_column].astype(str).apply(len)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Sentiment distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sentiment_counts = df['sentiment'].value_counts()
        ax1.bar(sentiment_counts.index, sentiment_counts.values,
               color=[self.colors[sent] for sent in sentiment_counts.index],
               alpha=0.8, edgecolor='black')
        ax1.set_title('Sentiment Distribution', fontweight='bold')
        ax1.set_ylabel('Count')
        
        # 2. Rating distribution (if available)
        if 'Score' in df.columns:
            ax2 = fig.add_subplot(gs[0, 1])
            rating_counts = df['Score'].value_counts().sort_index()
            colors_ratings = [self.colors['negative'] if r <= 2 else 
                            self.colors['positive'] if r >= 4 else 
                            self.colors['neutral'] for r in rating_counts.index]
            ax2.bar(rating_counts.index, rating_counts.values,
                   color=colors_ratings, alpha=0.8, edgecolor='black')
            ax2.set_title('Star Rating Distribution', fontweight='bold')
            ax2.set_xlabel('Stars')
            ax2.set_ylabel('Count')
        
        # 3. Review length by sentiment
        ax3 = fig.add_subplot(gs[0, 2])
        df.boxplot(column='text_length', by='sentiment', ax=ax3,
                  patch_artist=True)
        ax3.set_title('Review Length by Sentiment', fontweight='bold')
        ax3.set_xlabel('Sentiment')
        ax3.set_ylabel('Length (characters)')
        plt.sca(ax3)
        plt.xticks(rotation=0)
        
        # 4. Length histogram
        ax4 = fig.add_subplot(gs[1, :])
        positive_lengths = df[df['sentiment'] == 'positive']['text_length']
        negative_lengths = df[df['sentiment'] == 'negative']['text_length']
        ax4.hist([positive_lengths, negative_lengths], bins=50, 
                label=['Positive', 'Negative'],
                color=[self.colors['positive'], self.colors['negative']],
                alpha=0.6, edgecolor='black')
        ax4.set_title('Review Length Distribution Comparison', fontweight='bold')
        ax4.set_xlabel('Length (characters)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        # 5-6. Top words for positive and negative
        for col, sentiment, color in [(0, 'positive', self.colors['positive']),
                                       (1, 'negative', self.colors['negative'])]:
            ax = fig.add_subplot(gs[2, col])
            
            # Collect words
            all_words = []
            for text in df[df['sentiment'] == sentiment][text_column].astype(str):
                all_words.extend(text.split())
            
            word_freq = Counter(all_words)
            top_words = word_freq.most_common(10)
            
            words = [word for word, _ in top_words]
            counts = [count for _, count in top_words]
            
            y_pos = np.arange(len(words))
            ax.barh(y_pos, counts, color=color, alpha=0.8, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, fontsize=9)
            ax.invert_yaxis()
            ax.set_title(f'Top 10 Words: {sentiment.capitalize()}', fontweight='bold')
            ax.set_xlabel('Frequency')
        
        # 7. Statistics summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        stats_text = f"""
        DATASET STATISTICS
        
        Total Reviews: {len(df):,}
        Positive: {len(df[df['sentiment']=='positive']):,}
        Negative: {len(df[df['sentiment']=='negative']):,}
        
        Avg Length: {df['text_length'].mean():.0f} chars
        Min Length: {df['text_length'].min():.0f} chars
        Max Length: {df['text_length'].max():.0f} chars
        
        Balance Ratio:
        {(len(df[df['sentiment']=='positive'])/len(df)*100):.1f}% Positive
        {(len(df[df['sentiment']=='negative'])/len(df)*100):.1f}% Negative
        """
        
        ax7.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
        
        fig.suptitle('Sentiment Analysis - Exploratory Data Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Exploratory dashboard saved to {save_path}")
        
        return fig


# Testing and demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("DATA VISUALIZER MODULE")
    logger.info("\nAvailable visualization functions:")
    logger.info("  - plot_sentiment_distribution()")
    logger.info("  - generate_wordclouds()")
    logger.info("  - plot_review_length_distribution()")
    logger.info("  - plot_top_words()")
    logger.info("  - plot_rating_distribution()")
    logger.info("  - create_exploratory_dashboard()")
    logger.info("\nUsage example:")
    logger.info("  from visualizations.data_viz import DataVisualizer")
    logger.info("  viz = DataVisualizer()")
    logger.info("  viz.plot_sentiment_distribution(df, save_path='plots/sentiment_dist.png')")
    logger.info("  viz.create_exploratory_dashboard(df, save_path='plots/dashboard.png')")