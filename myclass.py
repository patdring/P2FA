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
    # classifier / preselection / preselect
    def getLeastSquareDeviations(self, trainingData, idealData):     
        #valid_indicies = idealData.x.combine_first(trainingData.x).values
        #print(valid_indicies)
        
        #if valid_indicies.shape[0] < trainingData.x.shape[0]:
        #   raise LittleUsableDataError("Only {} indicies are existing in both sets!".format(valid_indicies[0].shape[0]))

        lses = pd.DataFrame(None, columns = trainingData.columns[1:], index=idealData.columns[1:])
        greatestDeviations = pd.DataFrame(None, columns = trainingData.columns[1:], index=idealData.columns[1:]) 
        minLses = {}
        
        for ct in trainingData.columns[1:]:  
            #print("ct.x {}".format(trainingData[ct])) 
            for ci in idealData.columns[1:]:
                #print("\tci.x {}".format(idealData[ci])) 
                y_deviation = abs(idealData[ci] - trainingData[ct])
                greatestDeviations[ct][ci] = np.max(y_deviation)               
                lses[ct][ci] = (y_deviation ** 2).sum()              
            minLses[ct] = lses.sort_values(by=[ct])[ct].index[0]
      
        return minLses, greatestDeviations

    # selection / select
    def calcLinearRegression(self, testData, idealData, minLses, greatestDeviations):
        testLength = testData.x.values.shape[0]
        idealLength = idealData.x.values.shape[0]
        print("testData.x.values.shape[0] {}".format(testLength))
        print("idealData.x.values.shape[0] {}".format(idealLength))
        
        if testData.shape[1] != 2:
            raise TestDataShapeError("testData.shape[1]: {} is not 2!".format(testData.shape[1]))
        
        x_ideal = idealData.x.values.reshape(idealLength, 1)
        x_test = testData.x.values.reshape(testLength, 1)
        y_test = testData.y.values.reshape(testLength, 1)

        # replace with https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html#sphx-glr-auto-examples-linear-model-plot-robust-fit-py
        regr = linear_model.LinearRegression()

        testData = testData.assign(d='-')
        testData = testData.assign(n='-')

        matchIdealFunc = None
        matchIdealSlope = None
        matchIdealIntercept = None
        criterion = math.sqrt(2)

        for ct in greatestDeviations.columns:   
            y_ideal = idealData[minLses[ct]].values.reshape(idealLength, 1)
            regr.fit(x_ideal, y_ideal)     
            y_pred = regr.predict(x_test) 
            y_deviation = abs(y_test - y_pred)
            y_deviationMax = np.max(y_deviation)

            crit_dev = abs(greatestDeviations[ct][minLses[ct]] - y_deviationMax)
            print ("T:{} I:{} {} {}".format(ct, minLses[ct], greatestDeviations[ct][minLses[ct]], y_deviationMax))
            
            #if crit_dev < criterion:
            if True:
                criterion = crit_dev
                testData = testData.assign(d=pd.DataFrame(y_deviation))
                testData = testData.assign(n=pd.DataFrame([minLses[ct].replace('y','n')]*testLength))
                matchIdealFunc = minLses[ct]
                matchIdealSlope = regr.coef_
                matchIdealIntercept = regr.intercept_

        resultTable = testData.set_index('x')
    
        return resultTable, matchIdealFunc, matchIdealSlope, matchIdealIntercept
