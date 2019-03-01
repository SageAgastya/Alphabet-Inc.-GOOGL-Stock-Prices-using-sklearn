import math
import pandas as pd
import quandl
import numpy as np
import pickle
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt


df = quandl.get("WIKI/GOOGL")
df = pd.DataFrame(df, columns=["Adj. Open", "Adj. Close", "Adj. High", "Adj. Low"])
print df.tail()
forecast_col = "Adj. Close"
df["HL_per"] = ((df["Adj. High"] - df["Adj. Low"]) * 100.0) / df["Adj. Low"]
df.fillna(-99999, inplace=True)  # -99999 acts as a outlier for most of the algorithms..there
# maybe lot of missing data in datasets so in order to not to compromise with the data ,it's used.

forecast_out = int(math.ceil(0.1 * len(df)))
df["label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)  # drops the row of nAN
print df.tail()

# using np_array to make it handy
# take X=features and y=labels and handle as an array
X = np.array(df.drop(["label"], 1))
y = np.array(df["label"])
X = preprocessing.scale(X)
# standardized scaling means mean=0 and variance =1  ...i.e. values are normally distributed and are centered about zero..
# if distribution is not normal or inconsistent variance means higher order feature will dominate the training time.....

# start the cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# cross_validation completed


# call the model name
# fit=train and score=test
# n_jobs is a measure of processor speed..n_jobs=no. of linear lines tracking the training data
# if dont give value for n_jobs i.e. keeping void arguement ...then by default n_jobs=1
# if n_jobs=-1...gives the max_possible jobs or linear lines tracking the training data ...
# it increases efficiency but put load over the processor
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
# let's save this trained model so that we can use it again...using pickle...
f = open("linearregression.pickle", "wb")  # important
pickle.dump(clf, f)  # clf saved in the same directory
f.close()
new_f = open("linearregression.pickle", "rb")
pickle.load(new_f)

accuracy = clf.score(X_test, y_test)
print accuracy

"""
 score -Returns the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
((y_true - y_pred) ** 2).sum() and v is the total sum of squares
 ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and
 it can be negative (because the model can be arbitrarily worse).
 A constant model that always predicts the expected value
of y, disregarding the input features, would get a R^2 score of 0.0."""
# accuracy or confidence is the accuracy of the stock_price of 10%(days) into the future...
# instead of linear regression we can also choose SVM
"""

clf=svm.SVR()           #it is classification not regression but since it can find best fit
                           line using svm concept of maximum distance ..it can be used here
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print accuracy
"""
# going back by the no. of days in forecast_out...to predict the 30 days
# that we hv already covered...we are not predicting for future days
# bcz we want to check the accuracy of the model...
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
forecast_set = clf.predict(X_lately)
print forecast_set, accuracy, forecast_out

df["forecast"] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['forecast'].plot()
plt.xlabel('Date')
plt.ylabel('price')
plt.legend(loc=4)
plt.show()



