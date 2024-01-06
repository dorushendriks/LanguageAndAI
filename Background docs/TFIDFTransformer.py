from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from DataPreprocessor import DataPreprocessor

class TFIDFTransformer:
    def __init__(self, max_features=1000):
        """
        Initializes the TFIDFTransformer with a specified number of maximum features.
        :param max_features: The maximum number of features to consider (default: 1000).
        """
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.max_features)
    
    def clean_and_tokenize(self, data, text_column):
        """
        Applies text cleaning and tokenization to the specified column of the DataFrame.
        :param data: The DataFrame containing the text data.
        :param text_column: The name of the column containing the text to be processed.
        :return: DataFrame with added columns for cleaned and tokenized text.
        """
        data['clean_post'] = data[text_column].apply(DataPreprocessor.clean_text)
        data['tokenized_post'] = data['clean_post'].apply(DataPreprocessor.tokenize)
        return data

    def fit_transform(self, data, text_column='clean_post'):
        """
        Fits the TF-IDF vectorizer to the data and transforms the text column.
        :param data: The DataFrame containing the text data.
        :param text_column: The name of the column containing the cleaned text.
        :return: DataFrame containing the TF-IDF vectors.
        """
        tfidf_vectors = self.tfidf_vectorizer.fit_transform(data[text_column])
        tfidf_df = pd.DataFrame(tfidf_vectors.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
        return tfidf_df

# Example usage
# data = ... # Load or prepare your DataFrame
# tfidf_transformer = TFIDFTransformer(max_features=1000)
# data = tfidf_transformer.clean_and_tokenize(data, text_column='post')
# tfidf_df = tfidf_transformer.fit_transform(data)
