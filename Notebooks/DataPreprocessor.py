import pandas as pd
import re
import nltk
nltk.download('punkt')
from sklearn.model_selection import train_test_split, StratifiedKFold

class DataPreprocessor:
    def __init__(self, data):
        """
        Initializes the DataPreprocessor with the provided dataset.
        :param data: The dataset to be preprocessed.
        """
        self.data = data

    def rename_columns(self, rename_dict):
        """
        Renames columns in the dataset.
        :param rename_dict: A dictionary mapping from old column names to new column names.
        """
        self.data.rename(columns=rename_dict, inplace=True)

    def add_birth_decade(self):
        """
        Adds a 'birth_decade' column to the dataset, grouping birth years into decades.
        """
        if 'birth_year' not in self.data.columns:
            raise ValueError("DataFrame must contain a 'birth_year' column")
        self.data['birth_decade'] = self.data['birth_year'] // 10 * 10

    def split_dataset(self, train_size=0.7, test_size=0.2, random_state=None):
        """
        Splits the dataset into training, testing, and validation sets, 
        ensuring each birth year is represented proportionally.
        :param train_size: Proportion of the dataset to include in the train split
        :param test_size: Proportion of the dataset to include in the test split
        :param random_state: Controls the shuffling applied to the data before applying the split
        :return: train_data, test_data, validation_data
        """
        if train_size + test_size >= 1.0:
            raise ValueError("train_size and test_size sum should be less than 1.0")

        train_data, temp_data = train_test_split(
            self.data, 
            train_size=train_size, 
            stratify=self.data['birth_decade'], 
            random_state=random_state
        )

        new_test_size = 0.5
        test_data, validation_data = train_test_split(
            temp_data, 
            test_size=new_test_size, 
            stratify=temp_data['birth_decade'], 
            random_state=random_state
        )

        return train_data, test_data, validation_data

    def stratified_kfold_split(self, n_splits=2):
        """
        Apply Stratified K-Fold Cross-Validation based on 'birth_decade'.
        :param n_splits: Number of splits for K-Fold Cross-Validation
        :return: train_folds, test_folds
        """
        required_columns = {"author_ID", "post", "birth_year", "birth_decade"}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(f"DataFrame must contain: {required_columns}")

        skf = StratifiedKFold(n_splits=n_splits)
        X = self.data.drop('birth_decade', axis=1)
        y = self.data['birth_decade']

        train_folds = []
        test_folds = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            train_fold = pd.concat([X_train, y_train], axis=1)
            test_fold = pd.concat([X_test, y_test], axis=1)
            train_folds.append(train_fold)
            test_folds.append(test_fold)

        return train_folds, test_folds

    @staticmethod
    def clean_text(text):
        """
        Cleans the text data.
        :param text: The text to be cleaned.
        :return: Cleaned text.
        """
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#','', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        return text

    @staticmethod
    def tokenize(text):
        """
        Tokenizes the text.
        :param text: The text to be tokenized.
        :return: List of tokens.
        """
        return nltk.word_tokenize(text)

# Example usage
# data = pd.read_csv("birth_year.csv")
# preprocessor = DataPreprocessor(data)
# preprocessor.rename_columns({'auhtor_ID': 'author_ID'})
# preprocessor.add_birth_decade()
# train_data, test_data, validation_data = preprocessor.split_dataset()
# train_folds, test_folds = preprocessor.stratified_kfold_split(n