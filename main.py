from tabulate import tabulate
from bokeh.io import show, save, output_file
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, DataTable, TableColumn, Panel, Tabs, Div, NumberFormatter

import numpy as np
import pandas as pd
import math
from sklearn.linear_model import (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import timeit

import unittests as ut
import database as db
import myclass as mc


class VisualizationError(Exception):
    """Raised when the input value is too large"""
    pass

#TODO add boxplot for canidate!?

def plotData0(table_data, title='sample title', text="""Sample HTML Text"""):
    source = ColumnDataSource(table_data)
    columns = []
    div = Div(text=text)

    dt_width = table_data.columns.shape[0]*50
    dt_height = table_data.index.shape[0]*30

    for c in table_data.columns:
        #print(c)
        #if 'x' in c or'y' in c or 'd' in c:
        #    columns.append(TableColumn(field=c, title=c, formatter=NumberFormatter(format="0.0")))
        #else:
        columns.append(TableColumn(field=c, title=c))

    data_table = DataTable(source=source, columns=columns, height=dt_height, width=dt_width, index_position=None, sizing_mode = "stretch_width")
    layout = column(div, data_table) 

    return Panel(child=layout, title=title)

def plotData1(idealData, testData, matchIdealFunc, matchIdealSlope, matchIdealIntercept, title='title', text="""Sample HTML Text"""): 
    #if matchIdealFunc == None or matchIdealSlope == None or matchIdealIntercept == None:
    #    raise VisualizationError("")
    
    p = figure(title = 'Plot1')
      
    p.scatter('x','y',source=testData, fill_alpha=0.5, size=10, color='blue', legend_label='testData')
    p.scatter('x', 'y21', source=idealData, fill_alpha=0.5, size=10, color='green', legend_label='idealData')
    
    length = testData.x.values.shape[0]   
    step_size = 1
    x_values = np.arange(np.min(testData['x']), np.max(testData['x'])+1, step_size) 
    y_values = matchIdealSlope*x_values + matchIdealIntercept
    p.line(x_values, y_values[0], line_alpha=0.5, line_width=2, color='red', legend_label='Regression for ideal {} y={}*x+{}'.format(matchIdealFunc, matchIdealSlope[0][0], matchIdealIntercept[0]))
    div = Div(text=text,width=200, height=100)
    layout = column(div, p)

    return Panel(child=layout, title=title)

def plotData2(idealData, testData, matchIdealFunc, title='title', text="""Sample HTML Text"""):
    #if matchIdealFunc == None:
    #    raise VisualizationError("")

    p = figure(title = 'Plot2')
    p.scatter('x','y',source=testData, fill_alpha=0.5, size=10, color='blue', legend_label='testData')

    length_test = testData.x.values.shape[0]
    length_ideal = idealData.x.values.shape[0]
    
    x_ideal = idealData.x.values
    x_test = testData.x.values
    y_test = testData.y.values
        
    x_ideal = x_ideal.reshape(length_ideal, 1)
    x_test = x_test.reshape(length_test, 1)
    y_test = y_test.reshape(length_test, 1)

    #regr = linear_model.LinearRegression()
    regr = make_pipeline(PolynomialFeatures(3), LinearRegression())
    col_idealData = idealData.columns.values

    for ci in col_idealData[1:]: 
        step_size = 0.1
        x_values = np.arange(np.min(testData['x']), np.max(testData['x']), step_size)
        x_values = x_values.reshape(-1, 1)

        y_ideal = idealData[ci].values   
        y_ideal = y_ideal.reshape(length_ideal, 1)
        regr.fit(x_ideal, y_ideal)     
        y_values = regr.predict(x_values)

        x_values = x_values.T
        y_values = y_values.T 

        if matchIdealFunc == ci:
            p.line(x_values[0], y_values[0], line_alpha=0.5, line_width=2, color='red', legend_label='Regression')
        else:
            p.line(x_values[0], y_values[0], line_alpha=0.75, line_width=2, color='lightgray', legend_label='.')

    div = Div(text=text,width=200, height=100)
    layout = column(div, p)

    return Panel(child=layout, title=title)
 
def main():
    try:
        #trainingData = db.TrainingData(['Table1_0_training.csv', 'Table1_1_training.csv', 'Table1_2_training.csv', 'Table1_3_training.csv'], 'training_data')
        #idealData = db.IdealData('Table2_ideal.csv', 'ideal_data')
        #testData = db.TestData('Table3_test.csv', 'test_data')
        #resultData = db.ResultData('result_data')

        #df_trainingData = trainingData.readDataFromDB()
        #df_idealData = idealData.readDataFromDB()
        #df_testData = testData.readDataFromDB()

        
        df_trainingData = pd.read_csv("train.csv",delimiter = ',')
        df_idealData = pd.read_csv("ideal.csv",delimiter = ',')
        df_testData = pd.read_csv("test.csv",delimiter = ',')

        df_testData = df_testData.sort_values(by='x')

        vis_tabs = []
        vis_tabs.append(plotData0(df_idealData, 'ideal data'))
        vis_tabs.append(plotData0(df_trainingData, 'training data'))
        vis_tabs.append(plotData0(df_testData, 'test data'))

        #print("Training Data (raw)\n"+tabulate(df_trainingData, headers='keys', tablefmt='psql'))
        
        #print("\nIdeal Data (raw)\n"+tabulate(df_idealData.loc[:, :'y10'], headers='keys', tablefmt='psql'))
        #print("\nIdeal Data (raw)\n"+tabulate(df_idealData.loc[:, 'y11':'y20'], headers='keys', tablefmt='psql'))
        #print("\nIdeal Data (raw)\n"+tabulate(df_idealData.loc[:, 'y21':'y30'], headers='keys', tablefmt='psql'))
        #print("\nIdeal Data (raw)\n"+tabulate(df_idealData.loc[:, 'y31':'y40'], headers='keys', tablefmt='psql'))
        #print("\nIdeal Data (raw)\n"+tabulate(df_idealData.loc[:, 'y41':'y50'], headers='keys', tablefmt='psql'))
        
        #print("\nTest Data (raw)\n"+tabulate(df_testData, headers='keys', tablefmt='psql'))
       
        x = mc.MyClass()
        minLses, greatestDeviations = x.getLeastSquareDeviations(df_trainingData, df_idealData)
        vis_tabs.append(plotData0(greatestDeviations.reset_index(), 'Greatest Deviations'))
              
        #print("\nGreatest Deviations\n"+tabulate(greatestDeviations, headers='keys', tablefmt='psql'))
        print("Function assignments (Training:Ideal)\n{}".format(minLses))
        
        resultTable, matchIdealFunc, matchIdealSlope, matchIdealIntercept = x.calcLinearRegression(df_testData, df_idealData, minLses, greatestDeviations)

        #vis_tabs.append(plotData1(df_idealData, df_testData, matchIdealFunc, matchIdealSlope, matchIdealIntercept))
        vis_tabs.append(plotData2(df_idealData, df_testData, matchIdealFunc))
        
        # #write to sql database
        # #resultData.writeDataToDB(resultTable) 
        # #df_resultData = resultData.readDataFromDB()
        # df_resultData = resultTable
        # vis_tabs.append(plotData0(df_resultData, 'Result Data'))
        # print("Result Table\n"+tabulate(df_resultData, headers='keys', tablefmt='psql'))
        show(Tabs(tabs=vis_tabs))

    except mc.LittleUsableDataError as e:
        print("Data is not usable! ")
        details = e.args[0]
        print(e)

    except mc.TestDataShapeError as e:
        print("Data has not the expected shape! ")
        details = e.args[0]
        print(e)

    except VisualizationError as e:
        print("Data to visualize does not exist! ")
        details = e.args[0]
        print(e)
    
    except Exception as e:
        details = e.args[0]
        print(e)
    finally:
        print("End")
 
if __name__ == "__main__":
    # TODO commandline parameter (too much parameters ?!)
    print("Runtime: {}ms".format(timeit.timeit(main, number=1)))
    ut.unittest.main()
