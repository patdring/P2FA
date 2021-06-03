import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class Point2FunctionAllocatorError(Exception):
    '''Base class for exceptions in this module.'''
    pass


class UsableDataError(Point2FunctionAllocatorError):
    '''Raised when the input data is not usable'''
    pass


class TableDataShapeError(Point2FunctionAllocatorError):
    '''Raised when the input datas` shape is too wrong'''
    pass

class CPoint2FunctionAllocator:
    '''
    A class with methods uses training data to choose  ideal functions which 
    are the best fit out of the provided. Also test data provided to determine
    for each and every x-y-pair of values whether or not they can be assigned 
    to the chosen ideal functions.

    Only public attributes and methods are mentioned!

    Attributes:
            -

    Methods:
            preselectFunctions(trainingData, idealData):
                    Choosing the ideal functions for the training functions
            mapPoints2Functions(testData, idealData, matches, greatestDeviations):
                    Mapping the individual test case to the ideal functions
    '''

    def preselectFunctions(self, trainingData, idealData):
        '''
         Choosing the ideal functions for the training functions by how they 
         minimize the sum of all y-deviations squared (Least-Square)

            Parameters:
                trainingData (pandas.DataFrame): A frame with one column for x
                                                 and several for y values
                                                 representing training functions
                idealData (pandas.DataFrame): A frame with one column for x and
                                              several for y values representing
                                              ideal functions

            Returns:
                matches (dict): A directory which assigns the training 
                                functions selected by min. least-square to 
                                ideal functions {T:I, ...}
                greatesDeviations (pandas.DataFrame): A table which shows the 
                                                      largest deviation between
                                                      training (column) and 
                                                      ideal (row) function
        '''

        if trainingData.shape[0] < idealData.shape[0]:
           raise UsableDataError('Different x-values length,\
                                  mapping not possible!'
                                  .format(valid_indicies[0].shape[0]))

        lses = pd.DataFrame(None,
                            columns=trainingData.columns[1:],
                            index=idealData.columns[1:])
        greatestDeviations = pd.DataFrame(None,
                                          columns=trainingData.columns[1:],
                                          index=idealData.columns[1:])
        matches = {}

        for ct in trainingData.columns[1:]:
            for ci in idealData.columns[1:]:
                y_deviation = abs(idealData[ci] - trainingData[ct])
                greatestDeviations[ct][ci] = np.max(y_deviation)
                lses[ct][ci] = (y_deviation**2).sum()
            matches[ct] = lses.sort_values(by=[ct])[ct].index[0]

        return matches, greatestDeviations

    def mapPoints2Functions(self, testData, idealData, matches,
                            greatestDeviations):
        '''
        Mapping the individual test case to the ideal functions is that
        the existing maximum deviation of the calculated regression does not 
        exceed the largest deviation between training dataset and the ideal 
        function chosen for it by more than factor sqrt(2)

            Parameters:
                testData (pandas.DataFrame): A frame with x,y values
                                             representing test data
                idealData (pandas.DataFrame): A frame with one column for x and
                                              several for y values representing
                                              ideal functions
                matches (dict): A directory which assigns the training 
                                functions selected by min. least-square to 
                                ideal functions {T:I, ...}
                greatesDeviations (pandas.DataFrame): A table which shows the 
                                                      largest deviation between
                                                      training (column) and 
                                                      ideal (row) function

            Returns:
                df_resultTable (pandas.DataFrame): x-y pairs of the 
                                                   test data with its mapping 
                                                   to a ideal function and
                                                   corresponding indication of 
                                                   the deviation
        '''

        if testData.shape[1] != 2:
            raise TableDataShapeError(
                "testData.shape[1]: {} is not 2!".format(testData.shape[1]))

        df = pd.DataFrame(None, columns=['x', 'y', 'yd', 'n'])
        df_resultTable = pd.DataFrame(None, columns=['x', 'y', 'yd', 'n'])

        x_ideal = idealData.x.values.reshape(-1, 1)
        x_test = testData.x.values.reshape(-1, 1)
        y_test = testData.y.values.reshape(-1, 1)

        model = make_pipeline(PolynomialFeatures(3), LinearRegression())
        crit_factor = math.sqrt(2)

        for ct in greatestDeviations.columns:
            y_ideal = idealData[matches[ct]].values.reshape(-1, 1)

            model.fit(x_ideal, y_ideal)
            y_pred = model.predict(x_test)

            df['x'] = x_test.T[0]
            df['y'] = y_test.T[0]
            df['yd'] = abs(y_test - y_pred).T[0]
            df['n'] = matches[ct]

            df_resultTable = df_resultTable.append(
                df.loc[df['yd'] <= greatestDeviations[ct][matches[ct]] *
                       crit_factor])

        return df_resultTable