# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

dataset = pd.read_csv('collegePlace.csv')

dataset['Gender'] = dataset['Gender'].apply({'Male':0, 'Female':1}.get)

dataset.drop(['Stream'],axis = 1, inplace =True)

X = dataset.drop(["PlacedOrNot"] , axis=1)
Y = dataset.PlacedOrNot


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=47) 
nb = GaussianNB()
nb.fit(X_train , Y_train)

pickle.dump(nb, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


y_pred_nb = nb.predict(X_test)


from sklearn.metrics import accuracy_score

accuracy_score(y_pred_nb , Y_test)
