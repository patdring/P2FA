import unittest
import pandas as pd
import p2f_alloc as p2fa


class TestMethods(unittest.TestCase):
    """
    A class to represent a person.

    ...

    Attributes
    ----------
    name : str
        first name of the person
    surname : str
        family name of the person
    age : int
        age of the person

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    _df_testData = pd.DataFrame({
        "x": [-2, -1, 0, 1, 2],
        "y": [1, 2.1, 3.3, 4.4, 5.5],
    })

    _df_trainingData = pd.DataFrame({
        "x": [-2, -1, 0, 1, 2],
        "y1": [1, 2, 3, 4, 6],
        "y2": [2, 2, 0, -1, -2],
        "y3": [4, 2, 0, 2, 4],
        "y4": [2, 0, 2, 4, 8],
    })

    _df_idealData = pd.DataFrame({
        "x": [-2, -1, 0, 1, 2],
        "y1": [-2, -1, 0, 1, 2],
        "y2": [2, 1, 0, -1, -2],
        "y3": [4, 2, 0, 2, 4],
        "y4": [2, 0, 2, 4, 8],
        "y5": [9, 4, 1, -2, 1],
        "y6": [8, 7, 5, 11, 2],
        "y7": [2, 2, 3, 4, 6],
    })

    p2f_alloc = p2fa.CPoint2FunctionAllocator()

    def test_1(self):
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

        matches, greatestDeviations = self.p2f_alloc.preselectFunctions(
            self._df_trainingData, self._df_idealData)

        self.assertEqual(matches['y1'], 'y7')
        self.assertEqual(matches['y2'], 'y2')
        self.assertEqual(matches['y3'], 'y3')
        self.assertEqual(matches['y4'], 'y4')

        self.assertEqual(greatestDeviations['y1']['y5'], 8)
        self.assertEqual(greatestDeviations['y2']['y6'], 12)
        self.assertEqual(greatestDeviations['y3']['y7'], 3)
        self.assertEqual(greatestDeviations['y4']['y1'], 6)

    def test_2(self):
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

        matches, greatestDeviations = self.p2f_alloc.preselectFunctions(
            self._df_trainingData, self._df_idealData)
        resultTable = self.p2f_alloc.mapPoints2Functions(
            self._df_testData, self._df_idealData, matches, greatestDeviations)

        self.assertEqual(resultTable.iloc[0]['y'], 1.0)
        self.assertEqual(resultTable.iloc[0]['yd'], 0.9714285714285709)
        self.assertEqual(resultTable.iloc[0]['n'], 'y7')

        self.assertEqual(resultTable.iloc[2]['y'], 3.3)
        self.assertEqual(resultTable.iloc[2]['yd'], 0.47142857142857064)
        self.assertEqual(resultTable.iloc[2]['n'], 'y7')

        self.assertEqual(resultTable.iloc[4]['y'], 5.5)
        self.assertEqual(resultTable.iloc[4]['yd'], 0.47142857142857064)
        self.assertEqual(resultTable.iloc[4]['n'], 'y7')

        self.assertEqual(resultTable.iloc[6]['y'], 2.1)
        self.assertEqual(resultTable.iloc[6]['yd'], 1.1000000000000008)
        self.assertEqual(resultTable.iloc[6]['n'], 'y2')
