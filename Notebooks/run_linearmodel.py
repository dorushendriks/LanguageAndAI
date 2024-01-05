# non class based version
# class based version not merged yet because its not compatible with the preprocessing others did
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


df = pd.read_csv("../Data/birth_year.csv")




tfidf_vectorizer = TfidfVectorizer(max_features=2000)


X_train, X_test, y_train, y_test = train_test_split(df['post'], df['birth_year'], random_state=2024)

# Create a TF-IDF vectorizer and transform the text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# create and train
model = LinearRegression()
model.fit(X_train_tfidf, y_train)

# test and eval
predictions = model.predict(X_test_tfidf)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# see coefficients
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_

# extract them
coef_df = pd.DataFrame({'Word': feature_names, 'Coefficient': coefficients})


coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# show top 10
print(coef_df.head(10))