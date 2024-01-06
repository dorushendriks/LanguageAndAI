import DataPreprocessor

from TFIDFTransformer import TFIDFTransformer
from DataVisualizer import DataVisualizer

class DataPreparation:
    def __init__(self, data, max_features=1000):
        """
        Initializes the MainAnalysis with the provided dataset.
        :param data: The dataset to be analyzed.
        :param max_features: The maximum number of features for TF-IDF vectorization.
        """
        self.data = data
        self.preprocessor = DataPreprocessor.DataPreprocessor(data)
        self.tfidf_transformer = TFIDFTransformer(max_features)
        self.visualizer = DataVisualizer()

    def run(self):
        """
        Runs the main analysis workflow.
        """
        # Preprocessing steps
        self.preprocessor.rename_columns({'auhtor_ID': 'author_ID'})
        self.preprocessor.add_birth_decade()

        # # Visualize the distribution of birth years and decades
        # self.visualizer.visualize_birth_year_distribution(self.data)
        # self.visualizer.visualize_birth_decade_distribution(self.data)

        # Splitting the dataset
        train_data, test_data, validation_data = self.preprocessor.split_dataset()
        # self.visualizer.visualize_splits_by_decade(train_data, test_data, validation_data)
        # self.visualizer.visualize_proportional_splits_by_decade(train_data, test_data, validation_data)

        # # Stratified K-Fold Split
        # train_folds, test_folds = self.preprocessor.stratified_kfold_split()
        # self.visualizer.visualize_stratified_kfold_splits(train_folds, test_folds)

        # Clean and tokenize the 'post' column, and apply TF-IDF transformation
        print('Transforming the data')
        train_data = self.tfidf_transformer.clean_and_tokenize(train_data, 'post')
        self.X_train = self.tfidf_transformer.fit_transform(train_data)
        self.y_train = train_data.birth_year

        test_data = self.tfidf_transformer.clean_and_tokenize(test_data, 'post')
        self.X_test = self.tfidf_transformer.transform(test_data)
        self.y_test = test_data.birth_year

        # Visualize top words by TF-IDF scores
        # self.visualizer.visualize_top_words(tfidf_df)
        return self

    def get_training_data(self):
        return self.X_train, self.y_train
    
    def get_test_data(self):
        return self.X_test, self.y_test
    
    def get_feature_names_out(self):
        return self.tfidf_transformer.tfidf_vectorizer.get_feature_names_out()
    
# Example usage
# data = pd.read_csv("birth_year.csv")
# analysis = MainAnalysis(data, max_features=1000)
# analysis.run()

