from DataPreparation import *
from ModelCollection import *

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


class Analysis():
    def __init__(self, data, max_features:int=1000) -> None:
        '''
        Initialiser for the Analysis class
        It initializes a DataPreparation object and a ModelCollection object
        :param data:            A dataframe with reddit posts and author birth years
        :param max_features:    The maximum number of features that are included in the vocabulary
        '''
        print('Begin analysis')
        self.data = DataPreparation(data, max_features).run() # should then have preprocessed self.data split into train and test
        self.modelCollection = ModelCollection([DecisionTreeRegressor(max_features='log2'),
                                                RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1),
                                                SVR(kernel='linear', verbose=True),
                                                LinearRegression(n_jobs=-1)],
                                                self.data.get_feature_names_out()) 
        
    def run(self, n=100):
        '''
        Run the analysis by extracting the X and y values, fitting the models and evaluating the results
        :param n:   the number of words to consider as the most influential
        '''
        X_train, y_train = self.data.get_training_data()
        X_test, y_test = self.data.get_test_data()

        print('Training the models')
        self.modelCollection.fit(X_train, y_train)
        y_preds = self.modelCollection.predict(X_test)

        print('Evaluating')
        self.modelCollection.EvaluationMetrics(y_test, y_preds)
        self.modelCollection.overlapping_features(n)
        # self.modelCollection.visualize_feature_importances(n)


# Example usage
        
import time
data = pd.read_csv("./Data/birth_year.csv")
analysis = Analysis(data, max_features=1000)

start = time.time()
analysis.run()

time_ = time.time() - start
minutes = time_ // 60
seconds = time_ % 60
print(f'It took {minutes:.0f} minutes and {seconds:.0f} seconds to run')
