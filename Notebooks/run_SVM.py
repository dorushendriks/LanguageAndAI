# non class based version
# class based version not merged yet because its not compatible with the preprocessing others did
# takes longer than linear model, so print progress

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("../Data/birth_year.csv")

# same as linear model
X_train, X_test, y_train, y_test = train_test_split(df['post'], df['birth_year'], random_state=2024)


vectorizer = TfidfVectorizer(max_features=5000)
print("transform X 1/2")
X_train_tfidf = vectorizer.fit_transform(X_train)
print("transform X 2/2")
X_test_tfidf = vectorizer.transform(X_test)

print("fitting model")
model = SVR(kernel='linear', verbose=True) # it takes really long so verbose makes it print progress updates
model.fit(X_train_tfidf, y_train)

print("done fitting")
print("making predictions")
predictions = model.predict(X_test_tfidf)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# see coefficients
support_vectors = model.support_vectors_
coefficients = np.array(model.coef_.todense()).flatten()
coef_df = pd.DataFrame({'Word': vectorizer.get_feature_names_out(), 'Coefficient': coefficients})


coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
print(coef_df.head(10))

# results in
#           Word  Coefficient
#1933      gonna    21.564653
#2625       lmao    21.227812
#2213        idk    20.400130
#632         bro    17.062860
#1911       girl    16.563099
#4956         xd    16.094866
#897     college    15.442145
#3075       okay    15.243967
#3588  recommend    14.186508
#1913      girls    14.065566