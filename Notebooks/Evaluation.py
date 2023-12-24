from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns



class EvaluationComputations(): # can be inherited by the classes for the models maybe?
    """ 
    General idea I had is to use this class as a parent class for the model classes
    These methods should be the same for all of them
    I will further tune and test everything when data preprocessing and models are implemented etc :)  
    """

    def __init__(self) -> None:
        """
        Init can be overriden.
        The child class should have the property "self.model" which contains the sklearn model object
        """
        self.model = None  # to be implemented by child classes

    def EvaluationMetrics(self, y_true, y_pred):
        """ 
        Compute the evaluation metrics

        Args:
        y_true:     The birth_year values that you aim to predict (np.array?)
        y_pred:     The predicted birth_year values outputted by the model
        """
        self.r2 = r2_score(y_true, y_pred)
        self.mse = mean_squared_error(y_true, y_pred)
        self.rmse = mean_squared_error(y_true, y_pred, squared=False)
        self.mae = mean_absolute_error(y_true, y_pred)
        return self.r2, self.mse, self.rmse, self.mae
    
    def get_n_most_influential(self, n:int):
        """ 
        Compute the n most influential features and extract their names
        """
        match self.model.__class__.__name__:
            case 'LinearRegression' | 'SVR':
                feature_importances = self.model.coef_[:n]
                feature_names = self.feature_names_in_[:n]
            case 'DecisionTreeRegressor' | 'RandomForestRegressor':
                feature_importances = self.model.feature_importances_[:n]
                feature_names = self.model.feature_names_in_[:n]
        return feature_importances, feature_names
    
    def overlapping_features(self, n:int, other_model):
        """
        Compute how many and which features overlap between this model and another model

        Args:
        n:              The n most influential features to consider
        other_model:    Another (trained) model class object, of which the class should also inherit from this class  
        """
        _, feature_names = self.get_n_most_influential(n)
        _, feature_names_2 = other_model.get_n_most_influential(n)

        overlapping_features = set(feature_names).intersection( set(feature_names_2) )
        n_overlapping_features = len(overlapping_features)

        return overlapping_features, n_overlapping_features

    def visualize_feature_importances(self, n:int):
        """
        Create a bar plot that shows the feature importances
        This should further be adjusted when the actual models and data are available
        + made pretty if we want to use it in the report
        """
        importances, names = self.get_n_most_influential(n)
        sns.barplot(x=importances, y=names, orient='y')
        