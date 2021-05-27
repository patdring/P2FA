import numpy as np
import pandas as pd
import math
from sklearn.linear_model import (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

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
        matches = {}
        
        for ct in trainingData.columns[1:]:  
            #print("ct.x {}".format(trainingData[ct])) 
            for ci in idealData.columns[1:]:
                #print("\tci.x {}".format(idealData[ci])) 
                y_deviation = abs(idealData[ci] - trainingData[ct])
                greatestDeviations[ct][ci] = np.max(y_deviation)               
                lses[ct][ci] = (y_deviation ** 2).sum()              
            matches[ct] = lses.sort_values(by=[ct])[ct].index[0]
      
        return matches, greatestDeviations

    # selection / select
    def calcLinearRegression(self, testData, idealData, matches, greatestDeviations):       
        if testData.shape[1] != 2:
            raise TestDataShapeError("testData.shape[1]: {} is not 2!".format(testData.shape[1]))
        
        df = pd.DataFrame(None, columns = ['x', 'y', 'yp', 'yd', 'n'])
        

        x_ideal = idealData.x.values.reshape(-1, 1)
        x_test = testData.x.values.reshape(-1, 1)
        y_test = testData.y.values.reshape(-1, 1)

        # replace with https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html#sphx-glr-auto-examples-linear-model-plot-robust-fit-py
        model = make_pipeline(PolynomialFeatures(3), LinearRegression())

        #testData = testData.assign(d='-')
        #testData = testData.assign(n='-')

        for ct in greatestDeviations.columns:   
            y_ideal = idealData[matches[ct]].values.reshape(-1, 1)

            model.fit(x_ideal, y_ideal)
            y_pred  = model.predict(x_test)

            df['x'] = x_test.T[0]
            df['y'] = y_test.T[0]
            df['yp'] = y_pred.T[0]
            df['yd'] = abs(y_test - y_pred).T[0]
            df['n'] = matches[ct]
            #df.loc[df['yd'] == 'foo']

            print(df.loc[df['yd'] <= greatestDeviations[ct][matches[ct]]*math.sqrt(2)])

            #print(df)

            # y_deviation = abs(y_test - y_pred)
            # # TODO filter here!!!
            # #print(y_deviation)
            # y_deviationMax = np.max(y_deviation)

            # crit_dev = abs(greatestDeviations[ct][matches[ct]] - y_deviationMax)
            # print ("T:{} I:{} {} {}".format(ct, matches[ct], greatestDeviations[ct][matches[ct]], y_deviationMax))
            
            # #if crit_dev < criterion:
            # if True:
            #     criterion = crit_dev
            #     testData = testData.assign(d=pd.DataFrame(y_deviation))
            #     testData = testData.assign(n=pd.DataFrame([matches[ct].replace('y','n')]*testLength))
                
            #     matchIdealFunc = matches[ct]
            #     #matchIdealSlope = model.coef_
            #     #matchIdealIntercept = model.intercept_

            #     print("matchIdealFunc {}".format(matchIdealFunc))
            #     print("matchIdealSlope {}".format(matchIdealSlope))
            #     print("matchIdealIntercept {}".format(matchIdealIntercept))

        resultTable = testData.set_index('x')

        matchIdealFunc = None
        matchIdealSlope = None
        matchIdealIntercept = None
    
        return resultTable, matchIdealFunc, matchIdealSlope, matchIdealIntercept
