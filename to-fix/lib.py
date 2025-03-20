from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, pandas_udf
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

@pandas_udf("string")
def preprocess_text_udf(text_series: pd.Series) -> pd.Series:
    def preprocess(text):
        # Lowercase the text
        text = text.lower()
        
        # Remove special characters and extra spaces
        text = re.sub(r'\W+', ' ', text)
        
        # Split text into words
        words = text.split()
    
        # Remove stopwords
        words = [w for w in words if w not in stop_words]
    
        # Lemmatize words
        words = [lemmatizer.lemmatize(w) for w in words]
    
        out_text = ' '.join(words)
    
        return out_text

    return text_series.apply(preprocess)


@pandas_udf("float")
def sentiment_score_udf(text_series: pd.Series) -> pd.Series:
    return text_series.apply(lambda text: sia.polarity_scores(text)['compound'])


@pandas_udf("string")
def sentiment_label_udf(score_series: pd.Series) -> pd.Series:
    def label(score):
        if score > 0.05:
            return 'positive'
        elif score < -0.05:
            return 'negative'
        else:
            return 'neutral'
    return score_series.apply(label)


def preprocess(df, k):
    """
    Args:
        df (pySpark df)
        k (int): number of subreddits to consider

    Returns:
        df_avg (pySpark df): df with subreddit, avg sentiment score, and sentiment label
    """
    df_prep = df.drop('body', 'content', 'id', 'subreddit_id', 'title', 'author', 'content_len', 'summary', 'summary_len')

    df_top = df_prep.select('subreddit') \
                    .groupBy('subreddit').count() \
                    .sort('count', ascending=False) \
                    .limit(k)
    
    topk_subreddits = []
    [topk_subreddits.append(df_top.collect()[i][0]) for i in range(k)]
    
    df_topk = df_prep.filter(df_prep['subreddit'].isin(topk_subreddits))
    
    df_topk = df_topk.withColumn("clean_text", preprocess_text_udf(df_topk['normalizedBody']))
    
    df_vader = df_topk.withColumn("sentiment_score", sentiment_score_udf(df_topk['clean_text']))
    df_vader = df_vader.withColumn("sentiment_label", sentiment_label_udf(df_vader['sentiment_score']))
    
    df_avg = df_vader.groupBy("subreddit") \
                     .agg(F.round(F.avg("sentiment_score"), 4).alias("avg_sentiment_score")) \
                     .limit(k)
    
    df_avg = df_avg.withColumn("sentiment_label", sentiment_label_udf(df_avg['avg_sentiment_score']))

    return df_avg































    