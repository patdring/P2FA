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

    dt_width = table_data.columns.shape[0]*75
    dt_height = table_data.index.shape[0]*40

    for c in table_data.columns:
        #print(c)
        if 'x' in c or 'y' in c or 'd' in c:
            columns.append(TableColumn(field=c, title=c, formatter=NumberFormatter(format="0.00000")))
        else:
            columns.append(TableColumn(field=c, title=c))

    data_table = DataTable(source=source, columns=columns, height=dt_height, width=dt_width, index_position=None, sizing_mode = "stretch_both")
    layout = column(div, data_table) 

    return Panel(child=layout, title=title)

def plotData1(testData, resultData, title='title', text="""Sample HTML Text"""): 
    #if matchIdealFunc == None or matchIdealSlope == None or matchIdealIntercept == None:
    #    raise VisualizationError("")
    
    p = figure(title = 'Plot1')

    colors = ['red', 'green', 'blue', 'yellow']
    matches = resultData.n.unique()
    result = list(zip(matches.tolist(), colors))
      
    p.scatter('x','y',source=testData, fill_alpha=0.25, size=10, color='gray', legend_label='test Data')

    start_size = 12
    for m,c in result:
        p.scatter('x', 'y', source=resultData.loc[resultData['n'] == m], fill_alpha=0.75, size=start_size, color=c, legend_label=m)
        start_size -= 2
     
    div = Div(text=text,width=200, height=100)
    layout = column(div, p)

    return Panel(child=layout, title=title)

def plotData2(idealData, resultData, title='title', text="""Sample HTML Text"""):
    #if matchIdealFunc == None:
    #    raise VisualizationError("")

    p = figure(title = 'Plot2')

    colors = ['red', 'green', 'blue', 'yellow']
    matches = resultData.n.unique()
    result = list(zip(matches.tolist(), colors))

    x_ideal = idealData.x.values.reshape(-1, 1)
    step_size = 0.1
    x_values = np.arange(np.min(idealData['x']), np.max(idealData['x']), step_size)
    x_values = x_values.reshape(-1, 1)

    model = make_pipeline(PolynomialFeatures(3), LinearRegression())

    start_line_width = 6
    start_size = 12

    for m,c in result:
        y_ideal = idealData[m].values.reshape(-1, 1)
        model.fit(x_ideal, y_ideal)
        y_values = model.predict(x_values)
        p.line(x_values.flatten(), y_values.flatten(), line_alpha=0.5, line_width=start_line_width, color=c, legend_label=m)
        start_line_width -= 0.5
        p.scatter('x', 'y', source=resultData.loc[resultData['n'] == m], fill_alpha=0.75, size=start_size, color=c, marker = 'circle_dot', legend_label=m)
        start_size -= 2
  
    div = Div(text=text,width=200, height=100)
    layout = column(div, p)

    return Panel(child=layout, title=title)

def plotData3(idealData, resultData, trainingData, title='title', text="""Sample HTML Text"""):
    #if matchIdealFunc == None:
    #    raise VisualizationError("")

    p = figure(title = 'Plot3')

    train = trainingData.columns[1:]
    colors = ['red', 'green', 'blue', 'yellow']
    matches = resultData.n.unique()
    result = list(zip(train, matches.tolist(), colors))

    x_ideal = idealData.x.values.reshape(-1, 1)
    step_size = 0.1
    x_values = np.arange(np.min(idealData['x']), np.max(idealData['x']), step_size)
    x_values = x_values.reshape(-1, 1)

    model = make_pipeline(PolynomialFeatures(3), LinearRegression())

    for i in idealData.columns[1:]:
        y_ideal = idealData[i].values.reshape(-1, 1)
        model.fit(x_ideal, y_ideal)
        y_values = model.predict(x_values)
        if i in matches:
            p.line(x_values.flatten(), y_values.flatten(), line_alpha=1.0, line_width=2, color='black', legend_label='match')
        else:
            p.line(x_values.flatten(), y_values.flatten(), line_alpha=1.0, line_width=2, color='lightgray', legend_label='declined')   

    for t,m,c in result:
        #p.line(x_values.flatten(), y_values.flatten(), line_alpha=0.5, line_width=start_line_width, color=c, legend_label=m)
        p.scatter('x', t, source=trainingData, fill_alpha=1.0, size=9, color=c, marker = 'x', legend_label=t)
        p.scatter('x', m, source=idealData, fill_alpha=1.0, size=9, color=c, marker = 'cross', legend_label=m)
  
    div = Div(text=text,width=200, height=100)
    layout = column(div, p)

    return Panel(child=layout, title=title)
 
def main():
    try:
        trainingData = db.IdealData('train.csv', 'training_data')
        idealData = db.IdealData('ideal.csv', 'ideal_data')
        testData = db.TestData('test.csv', 'test_data')
        resultData = db.ResultData('result_data')

        df_trainingData = trainingData.readDataFromDB()
        df_idealData = idealData.readDataFromDB()
        df_testData = testData.readDataFromDB()
   
        #df_trainingData = pd.read_csv("train.csv",delimiter = ',')
        #df_idealData = pd.read_csv("ideal.csv",delimiter = ',')
        #df_testData = pd.read_csv("test.csv",delimiter = ',')

        df_testData = df_testData.sort_values(by='x')

        vis_tabs = []
        vis_tabs.append(plotData0(df_idealData, 'ideal data (raw)'))
        vis_tabs.append(plotData0(df_trainingData, 'training data (raw)'))
        vis_tabs.append(plotData0(df_testData, 'test data (x-sorted)'))

        #print("Training Data (raw)\n"+tabulate(df_trainingData, headers='keys', tablefmt='psql'))
        
        #print("\nIdeal Data (raw)\n"+tabulate(df_idealData.loc[:, :'y10'], headers='keys', tablefmt='psql'))
        #print("\nIdeal Data (raw)\n"+tabulate(df_idealData.loc[:, 'y11':'y20'], headers='keys', tablefmt='psql'))
        #print("\nIdeal Data (raw)\n"+tabulate(df_idealData.loc[:, 'y21':'y30'], headers='keys', tablefmt='psql'))
        #print("\nIdeal Data (raw)\n"+tabulate(df_idealData.loc[:, 'y31':'y40'], headers='keys', tablefmt='psql'))
        #print("\nIdeal Data (raw)\n"+tabulate(df_idealData.loc[:, 'y41':'y50'], headers='keys', tablefmt='psql'))
        
        #print("\nTest Data (raw)\n"+tabulate(df_testData, headers='keys', tablefmt='psql'))
       
        x = mc.MyClass()
        minLses, greatestDeviations = x.getLeastSquareDeviations(df_trainingData, df_idealData)
        greatestDeviations = greatestDeviations.rename_axis("I/T")
        vis_tabs.append(plotData0(greatestDeviations.reset_index(), 'greatest deviations'))
              
        #print("\nGreatest Deviations\n"+tabulate(greatestDeviations, headers='keys', tablefmt='psql'))
        #print("Function assignments (Training:Ideal)\n{}".format(minLses))
        
        df_resultTable = x.calcLinearRegression(df_testData, df_idealData, minLses, greatestDeviations)

        #vis_tabs.append(plotData0(df_resultTable, 'result table'))
        vis_tabs.append(plotData1(df_testData, df_resultTable))
        vis_tabs.append(plotData2(df_idealData, df_resultTable))
        vis_tabs.append(plotData3(df_idealData, df_resultTable, df_trainingData))
        
        # #write to sql database
        resultData.writeDataToDB(df_resultTable) 
        df_resultData = resultData.readDataFromDB()
        vis_tabs.append(plotData0(df_resultData, 'result data'))
        #print("Result Table\n"+tabulate(df_resultData, headers='keys', tablefmt='psql'))
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
