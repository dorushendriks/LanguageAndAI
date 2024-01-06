import pandas as pd
import matplotlib.pyplot as plt

class DataVisualizer:
    @staticmethod
    def visualize_birth_year_distribution(birth_year_df):
        """
        Visualizes the distribution of birth years.
        :param birth_year_df: DataFrame with a 'birth_year' column.
        """
        birth_year_frequency = (birth_year_df['birth_year'].value_counts()) / len(birth_year_df) * 100
        birth_year_frequency.sort_index().plot(kind='bar', figsize=(12, 6))
        plt.title('Birth Year Distribution')
        plt.xlabel('Birth Year')
        plt.ylabel('Percentage')
        plt.show()

    @staticmethod
    def visualize_birth_decade_distribution(birth_year_df):
        """
        Visualizes the distribution of birth decades.
        :param birth_year_df: DataFrame with a 'birth_decade' column.
        """
        birth_decade_frequency = (birth_year_df['birth_decade'].value_counts()) / len(birth_year_df) * 100
        birth_decade_frequency.sort_index().plot(kind='bar', figsize=(12, 6))
        plt.title('Birth Decade Distribution')
        plt.xlabel('Birth Decade')
        plt.ylabel('Percentage')
        plt.show()

    @staticmethod
    def visualize_splits_by_decade(train_data, test_data, validation_data):
        """
        Visualizes the distribution of birth decades in train, test, and validation datasets.
        :param train_data: Training dataset (DataFrame)
        :param test_data: Testing dataset (DataFrame)
        :param validation_data: Validation dataset (DataFrame)
        """
        train_counts = train_data['birth_decade'].value_counts().sort_index()
        test_counts = test_data['birth_decade'].value_counts().sort_index()
        val_counts = validation_data['birth_decade'].value_counts().sort_index()

        counts_df = pd.DataFrame({
            'Train': train_counts,
            'Test': test_counts,
            'Validation': val_counts
        }).fillna(0)

        counts_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Distribution of Birth Decades in Train, Test, and Validation Sets')
        plt.xlabel('Birth Decade')
        plt.ylabel('Count')
        plt.legend(title='Dataset')
        plt.show()

    @staticmethod
    def visualize_proportional_splits_by_decade(train_data, test_data, validation_data):
        """
        Visualizes the proportional distribution of birth decades in train, test, and validation datasets.
        :param train_data: Training dataset (DataFrame)
        :param test_data: Testing dataset (DataFrame)
        :param validation_data: Validation dataset (DataFrame)
        """
        train_counts = train_data['birth_decade'].value_counts().sort_index()
        test_counts = test_data['birth_decade'].value_counts().sort_index()
        val_counts = validation_data['birth_decade'].value_counts().sort_index()

        counts_df = pd.DataFrame({
            'Train': train_counts,
            'Test': test_counts,
            'Validation': val_counts
        }).fillna(0)

        total_counts = counts_df.sum(axis=1)
        proportional_counts = counts_df.divide(total_counts, axis=0)

        proportional_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Proportional Distribution of Birth Decades in Train, Test, and Validation Sets')
        plt.xlabel('Birth Decade')
        plt.ylabel('Proportion')
        plt.legend(title='Dataset')
        plt.show()

    @staticmethod
    def visualize_stratified_kfold_splits(train_folds, test_folds):
        """
        Visualizes the distribution of 'birth_decade' in each fold of the Stratified K-Fold splits.
        :param train_folds: List of training DataFrames
        :param test_folds: List of testing DataFrames
        """
        num_folds = len(train_folds)
        fig, axes = plt.subplots(num_folds, 1, figsize=(10, num_folds * 5))
        if num_folds == 1:
            axes = [axes]

        for i, (ax, train_fold, test_fold) in enumerate(zip(axes, train_folds, test_folds)):
            train_counts = train_fold['birth_decade'].value_counts().sort_index()
            test_counts = test_fold['birth_decade'].value_counts().sort_index()

            counts_df = pd.DataFrame({'Train': train_counts, 'Test': test_counts}).fillna(0)
            counts_df.plot(kind='bar', ax=ax)
            ax.set_title(f'Distribution in Fold {i + 1}')
            ax.set_xlabel('Birth Decade')
            ax.set_ylabel('Count')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_top_words(tfidf_df, top_n=20):
        """
        Visualizes the top words by summed TF-IDF scores.
        :param tfidf_df: DataFrame containing TF-IDF scores
        :param top_n: Number of top words to visualize
        """
        word_sum = tfidf_df.sum().sort_values(ascending=False)
        top_words = word_sum.head(top_n)

        plt.figure(figsize=(10, 6))
        top_words.plot(kind='bar')
        plt.title(f'Top {top_n} Words by Summed TF-IDF Scores')
        plt.ylabel('Summed TF-IDF Score')
        plt.xlabel('Words')
        plt.xticks(rotation=45)
        plt.show()

# Example usage of each method can be seen in the method's docstring.
