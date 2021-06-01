import numpy as np
import pandas as pd
import math
from sklearn.linear_model import (LinearRegression, TheilSenRegressor,
                                  RANSACRegressor, HuberRegressor)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class CPoint2FunctionAllocatorError(Exception):
    """Base class for exceptions in this module."""
    pass


class LittleUsableDataError(CPoint2FunctionAllocatorError):
    """Raised when the input value is too small"""
    pass


class CLineTableDataShapeError(CPoint2FunctionAllocatorError):
    """Raised when the input value is too large"""
    pass


# methods with underscore _ should be considered as private
class CPoint2FunctionAllocator:
    '''
    A class to represent a person.

    ...

    Attributes:
            -

    Methods:
            preselectFunctions(trainingData, idealData):
                    ...
            mapPoints2Functions(testData, idealData, matches, greatestDeviations):
                    ...
    '''

    def preselectFunctions(self, trainingData, idealData):
        '''
        Returns a bokeh panel with a bokeh table and a description text.

            Parameters:
                trainingData (pandas.DataFrame): A frame with one column for x
                                                 and several for y values
                                                 representing training functions
                idealData (pandas.DataFrame): A frame with one column for x and
                                              several for y values representing
                                              ideal functions

            Returns:
                Panel (bokeh.models.widgets.panels): Panel which can be
                added as a tab
        '''

        # if valid_indicies.shape[0] < trainingData.x.shape[0]:
        #   raise LittleUsableDataError("Only {} indicies are existing in both sets!".format(valid_indicies[0].shape[0]))

        lses = pd.DataFrame(None,
                            columns=trainingData.columns[1:],
                            index=idealData.columns[1:])
        greatestDeviations = pd.DataFrame(None,
                                          columns=trainingData.columns[1:],
                                          index=idealData.columns[1:])
        matches = {}

        for ct in trainingData.columns[1:]:
            #print("ct.x {}".format(trainingData[ct]))
            for ci in idealData.columns[1:]:
                #print("\tci.x {}".format(idealData[ci]))
                y_deviation = abs(idealData[ci] - trainingData[ct])
                greatestDeviations[ct][ci] = np.max(y_deviation)
                lses[ct][ci] = (y_deviation**2).sum()
            matches[ct] = lses.sort_values(by=[ct])[ct].index[0]

        return matches, greatestDeviations

    # selection / select
    def mapPoints2Functions(self, testData, idealData, matches,
                            greatestDeviations):
        '''
        Returns a bokeh panel with a bokeh table and a description text.

            Parameters:
                    testData (pandas.DataFrame): A frame with x,y values
                                                 representing test data
                    idealData (pandas.DataFrame): A frame with one column for x and
                                                  several for y values representing
                                                  ideal functions
                    matches (list): Todo
                    greatestDeviations (pandas.DataFrame): Todo

            Returns:
                Panel (bokeh.models.widgets.panels): Panel which can be
                added as a tab
        '''


        if testData.shape[1] != 2:
            raise CLineTableDataShapeError(
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
