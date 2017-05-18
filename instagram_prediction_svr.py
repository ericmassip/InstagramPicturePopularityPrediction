import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn import utils

dataframe = pd.read_csv('/Users/ericmassip/Desktop/TFG/Instagram_Data/allCategoriesTogetherFilteredByScoreOnly.csv')
#dataframe = dataframe[0:1100]
inputY = dataframe.loc[:, ['RangeOfLikes']].as_matrix()
inputY.ravel()
inputY = np.squeeze(inputY)
dataframe = dataframe.drop(['Filename', 'Likes', 'RangeOfLikes'], axis=1)
inputX = dataframe.loc[:, ['Posts','Followers','Following', 'Score']].as_matrix()

print(inputY)
print(inputX)

clf = SVR(kernel= 'rbf', C=1e4, gamma=0.01)
clf.fit(inputX, inputY)

accuracy = clf.score(inputX, inputY)
#print(accuracy)

#rowToPredict = [44066, 8388, 613, 0.709709]
#rowToPredict = np.reshape(rowToPredict, (1, -1))
#print(clf.predict(rowToPredict))