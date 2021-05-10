import numpy as np
import pandas as pd
import math
from sklearn import linear_model

class MyClassError(Exception):
    """Base class for exceptions in this module."""
    pass

class LittleUsableDataError(MyClassError):
    """Raised when the input value is too small"""
    pass

class TestDataShapeError(MyClassError):
    """Raised when the input value is too large"""
    pass

# methods with underscore _ should be considered as private
class MyClass:
    # classifier/ preselection / preselect
    def getLeastSquareDeviations(self, trainingData, idealData):     
        valid_indicies = idealData.X.combine_first(idealData.X).values
        
        if valid_indicies.shape[0] < trainingData.X.shape[0]:
           raise LittleUsableDataError("Only {} indicies are existing in both sets!".format(valid_indicies[0].shape[0]))

        lses = pd.DataFrame(None, columns = trainingData.columns[1:], index=idealData.columns[1:])
        greatestDeviations = pd.DataFrame(None, columns = trainingData.columns[1:], index=idealData.columns[1:]) 
        minLses = {}
        
        for ct in trainingData.columns[1:]:  
            for ci in idealData.columns[1:]:
                y_deviation = abs(idealData[ci].values[valid_indicies] - trainingData[ct].values[valid_indicies])
                greatestDeviations[ct][ci] = np.max(y_deviation)               
                lses[ct][ci] = (y_deviation ** 2).sum()              
            minLses[ct] = lses.sort_values(by=[ct])[ct].index[0]         
        return minLses, greatestDeviations

    # selection /select
    def calcLinearRegression(self, testData, idealData, minLses, greatestDeviations):
        length = testData.X.values.shape[0]
        
        if testData.shape[1] != 2:
            raise TestDataShapeError("testData.shape[1]: {} is not 2!".format(testData.shape[1]))
        
        x_ideal = idealData.X.values.reshape(length, 1)
        x_test = testData.X.values.reshape(length, 1)
        y_test = testData.Y.values.reshape(length, 1)

        regr = linear_model.LinearRegression()

        testData = testData.assign(D='-')
        testData = testData.assign(N='-')

        matchIdealFunc = None
        matchIdealSlope = None
        matchIdealIntercept = None
        criterion = math.sqrt(2)

        for ct in greatestDeviations.columns:   
            y_ideal = idealData[minLses[ct]].values.reshape(length, 1)
            regr.fit(x_ideal, y_ideal)     
            y_pred = regr.predict(x_test) 
            y_deviation = abs(y_test - y_pred)
            y_deviationMax = np.max(y_deviation)

            crit_dev = abs(greatestDeviations[ct][minLses[ct]] - y_deviationMax)
            print ("T:{} I:{} {}".format(ct, minLses[ct], crit_dev))
            
            if crit_dev < criterion:
                criterion = crit_dev
                testData = testData.assign(D=pd.DataFrame(y_deviation))
                testData = testData.assign(N=pd.DataFrame([minLses[ct].replace('Y','N')]*length))
                matchIdealFunc = minLses[ct]
                matchIdealSlope = regr.coef_
                matchIdealIntercept = regr.intercept_

        resultTable = testData.set_index('X')
    
        return resultTable, matchIdealFunc, matchIdealSlope, matchIdealIntercept
