from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from DataPreparation import * # includes visualizations, preprocessing, and tf-idf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import os

class ModelCollection():
    def __init__(self, models:list, feature_names) -> None:
        '''
        Initializer of the ModelCollection class
        :param models:          A list of sklearn model objects 
        :param feature_names:   The vocabulary extracted from the data
        '''
        self.models = models
        self.model_names = [model.__class__.__name__ for model in models]
        self.feature_names = feature_names # should be vectorizer.get_feature_names_out()

        if not os.path.exists('./Results'):
            os.makedirs('./Results')


    def fit(self, X_train, y_train):
        '''
        Fit the models in the model list
        :param X_train:     The training data
        :param y_train:     The true values corresponding to the training data
        '''
        for i, model in enumerate(self.models):
            model = model.fit(X_train, y_train)
            self.models[i] = model

    def predict(self, X_test):
        '''
        Predict on the provided data
        :param X_test:      The test data
        :return:            ndarray with the predicted birth_years
        '''
        self.y_preds = []
        for i, model in enumerate(self.models):
            y_pred = model.predict(X_test)
            self.y_preds.append(y_pred)
        return self.y_preds
    
    def EvaluationMetrics(self, y_true, y_preds):
        '''
        Compute the evaluation metrics r2, mse, rmse, and mae
        The results are stored in a CSV
        :param y_true:      The true birth_years
        :param y_preds:     list of four arrays with birth_year predictions for the different models
        :returns:           a dataframe with the evaluation metrics
        '''
        scores = {'r2':[], 'mse':[], 'rmse': [], 'mae': []}
        for y_pred in y_preds:
            scores['r2'].append(r2_score(y_true, y_pred))
            scores['mse'].append(mean_squared_error(y_true, y_pred))
            scores['rmse'].append(mean_squared_error(y_true, y_pred, squared=False))
            scores['mae'].append(mean_absolute_error(y_true, y_pred))

        scores_df = pd.DataFrame(scores, index=self.model_names)
        display(scores_df)
        scores_df.to_csv('Results/EvaluationMetrics.csv')
        return scores_df

    def get_n_most_influential(self, n:int):
        '''
        Get the n most influential words, these are saved to a CSV
        :param n:   The number of words to consider
        :returns:   A dataframe containing the most important words and their corresponding importances per model
        '''
        feature_importances_dict = {'coefficient':np.array([]), 'words':np.array([]), 'model':np.array([])}
        for model, name in zip(self.models, self.model_names):
            match name:
                case 'LinearRegressor':
                    feature_importances = np.absolute(model.coef_)
                case 'SVR':
                    feature_importances = np.absolute(model.coef_.flatten())
                case 'DecisionTreeRegressor' | 'RandomForestRegressor':
                    feature_importances = model.feature_importances_

            feature_importances_dict['coefficient'] = np.concatenate((feature_importances_dict['coefficient'], feature_importances), axis=0)
            feature_importances_dict['words']  = np.concatenate((feature_importances_dict['words'], self.feature_names), axis=0)
            feature_importances_dict['model']  = np.concatenate((feature_importances_dict['model'], [name]*len(feature_importances)), axis=0)

        feature_importances = pd.DataFrame(feature_importances_dict)
        feature_importances = feature_importances.sort_values(by='coefficient', ascending=False, ignore_index=True)
        feature_importances.to_csv('Results/feature_importances_per_model.csv')

        return feature_importances.iloc[:n,:]
    
    def overlapping_features(self, n:int):
        '''
        Determine which words overlap in importance for the different models and how many
        Both the overlapping words and the number of words are saved to separate CSVs
        :param n:   The number of words to consider
        :returns:   The two dataframes with overlapping words and the number of overlapping words
        '''
        feature_importances = self.get_n_most_influential(n)
        n_overlapping = pd.DataFrame(columns = self.model_names, index= self.model_names)
        overlapping = pd.DataFrame(columns = self.model_names, index= self.model_names)

        for name_1 in self.model_names:
            for name_2 in self.model_names:

                if name_1 == name_2:
                    continue
                feature_names_1 = feature_importances[feature_importances['model'] == name_1]['words']
                feature_names_2 = feature_importances[feature_importances['model'] == name_2]['words']

                overlapping_features_ = set(feature_names_1).intersection( set(feature_names_2) )
                n_overlapping_features = len(overlapping_features_)

                n_overlapping.loc[name_1, name_2] = n_overlapping.loc[name_2, name_1] = n_overlapping_features
                overlapping.loc[name_1, name_2] = overlapping.loc[name_2, name_1] = overlapping_features_
        
        n_overlapping.to_csv('Results/n_overlapping.csv')
        overlapping.to_csv('Results/overlapping.csv')
        return n_overlapping, overlapping

    def visualize_feature_importances(self, n:int):
        '''
        Create a bar chart that shows the n most important words and their relative importances
        :param n:   The number of words to consider
        '''
        for name in self.model_names:
            importances = self.get_n_most_influential(n) 
            importances_filtered = importances[importances['model'] == name]
            sns.barplot(data=importances_filtered, x='coefficient', y = 'words', orient='h')   
            plt.show()