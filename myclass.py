import numpy as np
import pandas as pd
import math
from sklearn.linear_model import (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
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
    """
    A class to represent a person.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    getLeastSquareDeviations(trainingData, idealData):
        ...
    calcLinearRegression(testData, idealData, matches, greatestDeviations):
        ...
    """
    # classifier / preselection / preselect
    def getLeastSquareDeviations(self, trainingData, idealData):     
        """
        Prints the person's name and age.

        If the argument 'additional' is passed, then it is appended after the main info.

        Parameters
        ----------
        additional : str, optional
            More info to be displayed (default is None)

        Returns
        -------
        None
        """

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
        """
        Prints the person's name and age.

        If the argument 'additional' is passed, then it is appended after the main info.

        Parameters
        ----------
        additional : str, optional
            More info to be displayed (default is None)

        Returns
        -------
        None
        """

        if testData.shape[1] != 2:
            raise TestDataShapeError("testData.shape[1]: {} is not 2!".format(testData.shape[1]))
        
        df = pd.DataFrame(None, columns = ['x', 'y', 'yp', 'yd', 'n'])
        df_resultTable = pd.DataFrame(None, columns = ['x', 'y', 'yp', 'yd', 'n'])
        
        x_ideal = idealData.x.values.reshape(-1, 1)
        x_test = testData.x.values.reshape(-1, 1)
        y_test = testData.y.values.reshape(-1, 1)

        model = make_pipeline(PolynomialFeatures(3), LinearRegression())

        for ct in greatestDeviations.columns:   
            y_ideal = idealData[matches[ct]].values.reshape(-1, 1)

            model.fit(x_ideal, y_ideal)
            y_pred  = model.predict(x_test)

            df['x'] = x_test.T[0]
            df['y'] = y_test.T[0]
            df['yp'] = y_pred.T[0]
            df['yd'] = abs(y_test - y_pred).T[0]
            df['n'] = matches[ct]

            df_resultTable = df_resultTable.append(df.loc[df['yd'] <= greatestDeviations[ct][matches[ct]]*math.sqrt(2)])

        return df_resultTable
