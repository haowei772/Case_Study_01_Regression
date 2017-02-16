import numpy as np
import pandas as pd
import scipy.stats as scs
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import xgboost as xgb
import math


#
# INPUT SPECIFC AREA
#

df = pd.read_csv('data/train.csv', low_memory=False, parse_dates=[9], infer_datetime_format=True)
df['SalesID'] = df['SalesID'].astype(int)
df['ProductGroup'] = df['ProductGroup'].astype('category')
df['fiModelDesc'] = df['fiModelDesc'].astype('category')

df['sale_year'] = df['saledate'].dt.year
df['sale_month'] = df['saledate'].dt.month
df['sale_day'] = df['saledate'].dt.day
df['sale_day_of_week'] = df['saledate'].dt.dayofweek

df['age'] = df['sale_year'] - df['YearMade']

df['saledate'] = df['saledate'].astype(int)
df['saledate'] = df['saledate'] / 1E11

y_colname = 'SalePrice'

#
# END Of data specifc area
#

# Generic area
input("what are the columns with least amount of nulls?")
length = len(df)
null_df = pd.DataFrame(sorted([[df[column].isnull().sum()/length,column] for column in df.columns]), columns=['pct_nulls', 'column'])
print(null_df)


#pct_threshold is the maximum % of nulls allowd
pct_threshold = 0.0
input("Removing columns that have nulls beyond threshold {}".format(pct_threshold))
df = df.loc[:,list(null_df[null_df['pct_nulls'] <= pct_threshold]['column'].values)]
print(df)

input("The data type for each column")
for column in df.columns:
    print(df[column].dtype,column)

# which columns are currently numeric?
numeric_idx = [True if df[column].dtype in ['int32','int64','float32','float64'] else False for column in df.columns]

# Selecting only the numeric data
X_df = df.iloc[:,numeric_idx]
y = df[y_colname]
del X_df[y_colname]

X = X_df.values

input("Pearson correlation for the training set")
plt.title('Pearson Correlation for training set')
sns.heatmap(X_df.corr(),
            linewidths=0.1,
            vmax=1.0,
            square=True,
            cmap="PuBuGn",
            linecolor='w',
            annot=False)

for column in X_df.columns:
    plt.hist(df[column], bins=20)
    #sns.stripplot(x=column, y='SalePrice', data=df, size=1, jitter=True);
    plt.title(column)
    plt.show()
    plt.clf()

input("histograms for each remaining column")
for column in X_df.columns:
    plt.hist(df[column], bins=20)
    #sns.stripplot(x=column, y='SalePrice', data=df, size=1, jitter=True);
    plt.title(column)
    plt.show()
    plt.clf()

input("Final training values")
print(X_df.info())


X = X_df.values
y = y.values
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size = 0.20)

input("About to run xgboost model")
model = xgb.XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.05, nthread=4)
print("Fitting model")
fit = model.fit(X_train, y_train, early_stopping_rounds=10,verbose=True)

print("Predicting on test set")
predictions = gbm.predict(X_test)

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

print(rmsle(y_test,y_pred))
