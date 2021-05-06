import unittest
import pandas as pd
import myclass as mc

class TestMethods(unittest.TestCase):

    _df_testData = pd.DataFrame(
        {
            "X": [-2, -1, 0, 1, 2],
            "Y": [1, 2.1, 3.3, 4.4, 5.5],
        }
    )

    _df_trainingData = pd.DataFrame(
        {
            "X": [-2, -1, 0, 1, 2],
            "Y1": [1, 2, 3, 4, 6],
            "Y2": [2, 2, 0, -1, -2],
            "Y3": [4, 2, 0, 2, 4],
            "Y4": [2, 0, 2, 4, 8],
        }
    )

    _df_idealData = pd.DataFrame(
        {
            "X": [-2, -1, 0, 1, 2],
            "Y1": [-2, -1, 0, 1, 2],
            "Y2": [2, 1, 0, -1, -2],
            "Y3": [4, 2, 0, 2, 4],
            "Y4": [2, 0, 2, 4, 8],
            "Y5": [9, 4, 1, -2, 1],
            "Y6": [8, 7, 5, 11, 2],
            "Y7": [2, 2, 3, 4, 6],
        }
    )

    x = mc.MyClass()

    def test_1(self):
        minLses, greatestDeviations = self.x.getLeastSquareDeviations(self._df_trainingData, self._df_idealData)

        self.assertEqual(minLses['Y1'], 'Y7')
        self.assertEqual(minLses['Y2'], 'Y2')
        self.assertEqual(minLses['Y3'], 'Y3')
        self.assertEqual(minLses['Y4'], 'Y4')

        self.assertEqual(greatestDeviations['Y1']['Y5'], 8)
        self.assertEqual(greatestDeviations['Y2']['Y6'], 12)
        self.assertEqual(greatestDeviations['Y3']['Y7'], 3)
        self.assertEqual(greatestDeviations['Y4']['Y1'], 6)

    def test_2(self):
        minLses, greatestDeviations = self.x.getLeastSquareDeviations(self._df_trainingData, self._df_idealData)
        resultTable, matchIdealFunc, matchIdealSlope, matchIdealIntercept = self.x.calcLinearRegression(self._df_testData, self._df_idealData, minLses, greatestDeviations)
        
        self.assertEqual(matchIdealFunc, 'Y7')
        self.assertEqual(matchIdealSlope, 1.0)
        self.assertEqual(matchIdealIntercept, 3.4)

        self.assertEqual(resultTable.iloc[0]['Y'], 1.0)
        self.assertEqual(resultTable.iloc[0]['N'], 'N7')
        self.assertEqual(resultTable.iloc[0]['D'], 0.3999999999999999)

        self.assertEqual(resultTable.iloc[1]['Y'], 2.1)
        self.assertEqual(resultTable.iloc[1]['N'], 'N7')
        self.assertEqual(resultTable.iloc[1]['D'], 0.2999999999999998)

        self.assertEqual(resultTable.iloc[2]['Y'], 3.3)
        self.assertEqual(resultTable.iloc[2]['N'], 'N7')
        self.assertEqual(resultTable.iloc[2]['D'], 0.10000000000000009)

        self.assertEqual(resultTable.iloc[3]['Y'], 4.4)
        self.assertEqual(resultTable.iloc[3]['N'], 'N7')
        self.assertEqual(resultTable.iloc[3]['D'], 0.0)

        self.assertEqual(resultTable.iloc[4]['Y'], 5.5)
        self.assertEqual(resultTable.iloc[4]['N'], 'N7')
        self.assertEqual(resultTable.iloc[4]['D'], 0.09999999999999964)

#TODO add unittest to catch exception