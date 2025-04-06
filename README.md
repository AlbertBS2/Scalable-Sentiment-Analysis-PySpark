## Scalable Reddit Sentiment Analysis with PySpark
This project presents a scalable pipeline for large-scale sentiment analysis on Reddit data using Python and PySpark. Utilizing the Webis-TLDR-17 corpus (3.8M Reddit posts in JSON), the pipeline ingests, preprocesses, and analyzes textual data to uncover sentiment trends across top subreddits. Preprocessing steps include tokenization, stopword removal, and lemmatization, enabling effective sentiment scoring via NLTK’s VADER analyzer. Scalability experiments were conducted on dataset subsets ranging from 50K to 500K posts, revealing that processing time increases linearly with dataset size, while performance significantly improves with additional computing nodes. The results confirm that distributed processing with PySpark is an effective solution for handling large social media datasets.

### The project is divided into three main files:

`partition-data.ipynb`: Ideated and used to partition the reddit dataset into smaller chunks to be uploaded to our own HDFS cluster.

`preprocessing-and-analysis.ipynb`: Preprocessing and top-25 subreddits sentiment analysis of a reddit dataset chunk.

`scalability-tests.ipynb`: Performed tests to compare compute time with different ressource allocation and dataset size.
