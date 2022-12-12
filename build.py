# import liberies

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


#read the file

df = pd.read_csv('deploy_df')


#drop un necessery row

df.drop('Unnamed: 0',axis=1,inplace=True)



x = x=df.drop('Price',axis=1)          #select x
print(x.head())


#selsct y

y=df['Price']


#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50)


#implementing CAT boost algorithm

from catboost import CatBoostRegressor

cat=CatBoostRegressor()
cat.fit(X_train,y_train)

y_predict=cat.predict(X_test)


#Use pickle to save our model so that we can use it later

import pickle

# Saving model

pickle.dump(cat, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(y_predict)
