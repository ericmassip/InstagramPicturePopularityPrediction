import csv
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn import preprocessing, utils
from sklearn.model_selection import KFold

dataframe = pd.read_csv('/Users/ericmassip/Desktop/TFG/Instagram_Data/allCategoriesTogetherFilteredByScoreOnly.csv')
#dataframe = dataframe[0:1100]
inputY = dataframe.loc[:, ['RangeOfLikes']].as_matrix()
inputY.ravel()
inputY = np.squeeze(inputY)
dataframe = dataframe.drop(['Filename', 'Likes', 'RangeOfLikes'], axis=1)
inputX = dataframe.loc[:, ['Posts','Followers','Following', 'Score']].as_matrix()

def get_score(x_train, y_train, x_test, y_test):
    regressionModel = SVR(kernel= 'rbf', C=2e3, gamma=0.0000000001)
    regressionModel.fit(x_train, y_train)
    score = regressionModel.score(x_test, y_test)
    return (score)

kf = KFold(n_splits=12, shuffle=False)
kf.get_n_splits(inputX)

crossValidationScore = 0

for train_index, test_index in kf.split(inputX):
    inputX_train, inputX_test = inputX[train_index], inputX[test_index]
    inputY_train, inputY_test = inputY[train_index], inputY[test_index]
    scoreForThisSplit = get_score(inputX_train, inputY_train, inputX_test, inputY_test)
    print("Accuracy = ", scoreForThisSplit)
    crossValidationScore = crossValidationScore + scoreForThisSplit

print("Cross Validation Score = ", crossValidationScore/12)