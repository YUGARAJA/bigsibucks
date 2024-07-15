import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Sample text data (replace with your actual data)
text_data = [
    "The three-language policy will promote cultural understanding.",
    "Many parents are concerned about the three-language system.",
    "I support the introduction of a three-language system in schools.",
    "Opposition parties criticize the government's three-language policy.",
    "Students will benefit from learning multiple languages.",
]

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Analyze sentiment for each text
sentiments = []
for text in text_data:
    sentiment_scores = sid.polarity_scores(text)
    sentiments.append(sentiment_scores)

# Plot sentiment distribution
labels = ['Negative', 'Neutral', 'Positive']
sentiment_counts = [sum(1 for s in sentiments if s['compound'] < -0.05),  # Negative sentiment
                    sum(1 for s in sentiments if -0.05 <= s['compound'] <= 0.05),  # Neutral sentiment
                    sum(1 for s in sentiments if s['compound'] > 0.05)]  # Positive sentiment

plt.bar(labels, sentiment_counts, color=['red', 'grey', 'green'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Analysis for Three-Language Policy')
plt.show()
