from tabulate import tabulate
from bokeh.io import show, save, output_file
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, DataTable, TableColumn, Panel,
                          Tabs, Div, NumberFormatter)
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import timeit
import sys
import getopt
import database as db
import p2f_alloc as p2fa


class VisualizationError(Exception):
    '''Raised when data connot be visualized'''
    pass


def createDataTablePanel(table_data,
                         title='sample title',
                         text='Sample HTML Text'):
    '''
    Returns a bokeh panel with a bokeh table and a description text.

            Parameters:
                    table_data (pandas.DataFrame): Table to display
                    title (str): Title text display describing tab
                    text (str): A [html formatted] description text

            Returns:
                    Panel (bokeh.models.widgets.panels): Panel which can be
                    added as a tab
    '''

    source = ColumnDataSource(table_data)
    columns = []
    div = Div(text=text)

    dt_width = table_data.columns.shape[0] * 75
    dt_height = table_data.index.shape[0] * 40

    for c in table_data.columns:
        if 'x' in c or 'y' in c or 'd' in c:
            columns.append(
                TableColumn(field=c,
                            title=c,
                            formatter=NumberFormatter(format='0.00000')))
        else:
            columns.append(TableColumn(field=c, title=c))

    data_table = DataTable(source=source,
                           columns=columns,
                           height=dt_height,
                           width=dt_width,
                           index_position=None,
                           sizing_mode='stretch_both')
    layout = column(div, data_table)

    return Panel(child=layout, title=title)


def createMatchingPointsPanel(testData,
                              resultData,
                              title='title',
                              text='Sample HTML Text'):
    '''
    Returns a bokeh panel with bokeh scatter plot and a description text.
    Scatter Plot displays:

        - Points from the test dataset mapped to ideal function also
          corresponding the condition y-deviation smaller than factor
          sqrt(2) are displayed/coloured

            Parameters:
                    testData (pandas.DataFrame): A frame with x,y values
                                                 representing test data
                    resultData (pandas.DataFrame): A frame with mapped x,y values
                                                   to an ideal function and the
                                                   existing deviation
                    title (str): Title text display describing tab
                    text (str): A [html formatted] description text

            Returns:
                    Panel (bokeh.models.widgets.panels): Panel which can be
                    added as a tab
    '''
    # if matchIdealFunc == None or matchIdealSlope == None or matchIdealIntercept == None:
    #    raise VisualizationError('')

    p = figure(title=title)

    colors = ['red', 'green', 'blue', 'yellow']
    matches = resultData.n.unique()
    result = list(zip(matches.tolist(), colors))

    p.scatter('x',
              'y',
              source=testData,
              fill_alpha=0.25,
              size=10,
              color='gray',
              legend_label='test Data')

    start_size = 12
    for m, c in result:
        p.scatter('x',
                  'y',
                  source=resultData.loc[resultData['n'] == m],
                  fill_alpha=0.75,
                  size=start_size,
                  color=c,
                  legend_label=m)
        start_size -= 2

    div = Div(text=text, width=200, height=100)
    layout = column(div, p)

    return Panel(child=layout, title=title)


def createRegressionPlotPanel(idealData,
                              resultData,
                              title='title',
                              text='Sample HTML Text'):
    '''
    Returns a bokeh panel with bokeh scatter plot,lines and a description text.
    Scatter Plot displays:

        - Matching points from test dataset and ideal functions
        - Regression lines assigned to the corresponding Ideal functions 
          points

            Parameters:
                    idealData (pandas.DataFrame): A frame with one column for x and
                                                  several for y values representing
                                                  ideal functions
                    resultData (pandas.DataFrame): A frame with mapped x,y values
                                                   to an ideal function and the
                                                   existing deviation
                    title (str): Title text display describing tab
                    text (str): A [html formatted] description text

            Returns:
                    Panel (bokeh.models.widgets.panels): Panel which can be
                    added as a tab
    '''

    # if matchIdealFunc == None:
    #    raise VisualizationError('')

    p = figure(title=title)

    colors = ['red', 'green', 'blue', 'yellow']
    matches = resultData.n.unique()
    result = list(zip(matches.tolist(), colors))

    x_ideal = idealData.x.values.reshape(-1, 1)
    step_size = 0.1
    x_values = np.arange(np.min(idealData['x']), np.max(idealData['x']),
                         step_size)
    x_values = x_values.reshape(-1, 1)

    model = make_pipeline(PolynomialFeatures(3), LinearRegression())

    start_line_width = 6
    start_size = 12

    for m, c in result:
        y_ideal = idealData[m].values.reshape(-1, 1)
        model.fit(x_ideal, y_ideal)
        y_values = model.predict(x_values)
        p.line(x_values.flatten(),
               y_values.flatten(),
               line_alpha=0.5,
               line_width=start_line_width,
               color=c,
               legend_label=m)
        start_line_width -= 0.5
        p.scatter('x',
                  'y',
                  source=resultData.loc[resultData['n'] == m],
                  fill_alpha=0.75,
                  size=start_size,
                  color=c,
                  marker='circle_dot',
                  legend_label=m)
        start_size -= 2

    div = Div(text=text, width=200, height=100)
    layout = column(div, p)

    return Panel(child=layout, title=title)


def createMappedPointsPanel(idealData,
                            resultData,
                            trainingData,
                            title='title',
                            text='Sample HTML Text'):
    '''
    Returns a bokeh panel with bokeh scatter plot,lines and a description text.
    Scatter Plot displays:

        - Visualization of points from training and ideal functions datasets
        - With bokeh built-in zoom function, deviations(test to ideal or to 
          regression lines) can be determined or displayed

        Parameters:
                idealData (pandas.DataFrame): A frame with one column for x and
                                              several for y values representing
                                              ideal functions
                resultData (pandas.DataFrame): A frame with mapped x,y values
                                               to an ideal function and the
                                               existing deviation
                trainingData (pandas.DataFrame): A frame with one column for x
                                                 and several for y values
                                                 representing training functions
                title (str): Title text display describing tab
                text (str): A [html formatted] description text

        Returns:
                Panel (bokeh.models.widgets.panels): Panel which can be added
                as a tab
    '''

    # if matchIdealFunc == None:
    #    raise VisualizationError('')

    p = figure(title=title)

    train = trainingData.columns[1:]
    colors = ['red', 'green', 'blue', 'yellow']
    matches = resultData.n.unique()
    result = list(zip(train, matches.tolist(), colors))

    x_ideal = idealData.x.values.reshape(-1, 1)
    step_size = 0.1
    x_values = np.arange(np.min(idealData['x']), np.max(idealData['x']),
                         step_size)
    x_values = x_values.reshape(-1, 1)

    model = make_pipeline(PolynomialFeatures(3), LinearRegression())

    for i in idealData.columns[1:]:
        y_ideal = idealData[i].values.reshape(-1, 1)
        model.fit(x_ideal, y_ideal)
        y_values = model.predict(x_values)
        if i in matches:
            p.line(x_values.flatten(),
                   y_values.flatten(),
                   line_alpha=1.0,
                   line_width=2,
                   color='black',
                   legend_label='match')
        else:
            p.line(x_values.flatten(),
                   y_values.flatten(),
                   line_alpha=1.0,
                   line_width=2,
                   color='lightgray',
                   legend_label='declined')

    for t, m, c in result:
        p.scatter('x',
                  t,
                  source=trainingData,
                  fill_alpha=1.0,
                  size=9,
                  color=c,
                  marker='x',
                  legend_label=t)
        p.scatter('x',
                  m,
                  source=idealData,
                  fill_alpha=1.0,
                  size=9,
                  color=c,
                  marker='cross',
                  legend_label=m)

    div = Div(text=text, width=200, height=100)
    layout = column(div, p)

    return Panel(child=layout, title=title)


def main(argv):
    '''
    The main function and entry point for the project. Gives help via command
    line and allows to pass the relevant CSV files via parameters. Also allows
    the activation of a verbose mode. Is responsible for the visualization of
    tables and graphs. Main functionality, searches for suitable candidates
    from ideal and training data by means of criterion and then maps them to
    test data by means of calculated regression and another criterion.

    1st criterion: Minimize the sum of all y-deviations squared (Least-Square)
    2nd criterion: Existing maximum deviation of the calculated regression does 
                   not exceed the largest deviation between training dataset 
                   and the ideal function chosen for it by more than factor 
                   sqrt(2)

        Parameters:
                argv (sys.argv): An array with command line parameters

        Returns:
                -
    '''

    try:
        idealfile = ''
        trainfile = ''
        testfile = ''
        verbose = False

        try:
            opts, args = getopt.getopt(argv, 'hi:t:e:v',
                                       ['ifile=', 'tfile=', 'efile='])
        except getopt.GetoptError:
            print('p2fa.py -i <idealfile> -t <trainfile> -e <testfile> [-v]')
            sys.exit(2)

        if not 'h' or 'v' in opts and len(opts) < 3:
            print('p2fa.py -i <idealfile> -t <trainfile> -e <testfile> [-v]')
            sys.exit(2)

        for opt, arg in opts:
            if opt == '-h':
                print(
                    'p2fa.py -i <idealfile> -t <trainfile> -e <testfile> [-v]')
                sys.exit()
            elif opt in ('-i', '--ifile'):
                idealfile = arg
            elif opt in ('-t', '--tfile'):
                trainfile = arg
            elif opt in ('-e', '--efile'):
                testfile = arg
            elif opt in ('-v', '--verbose'):
                verbose = True

        if idealfile == '' or trainfile == '' or testfile == '':
            print('p2fa.py -i <idealfile> -t <trainfile> -e <testfile> [-v]')
            sys.exit(2)

        trainingData = db.CBasicTableData(trainfile, 'training_data')
        idealData = db.CBasicTableData(idealfile, 'ideal_data')
        testData = db.CLineTableData(testfile, 'test_data')
        resultData = db.ResultData('result_data')

        df_trainingData = trainingData.readDataFromDB()
        df_idealData = idealData.readDataFromDB()
        df_testData = testData.readDataFromDB()

        #df_trainingData = pd.read_csv('train.csv',delimiter = ',')
        #df_idealData = pd.read_csv('ideal.csv',delimiter = ',')
        #df_testData = pd.read_csv('test.csv',delimiter = ',')

        df_testData = df_testData.sort_values(by='x')

        vis_tabs = []
        vis_tabs.append(
            createDataTablePanel(
                df_idealData, 'Ideal Data (raw)',
                '<b>Ideal functions’</b> database table read from file {}'.
                format(idealfile)))
        vis_tabs.append(
            createDataTablePanel(
                df_trainingData, 'Training Data (raw)',
                '<b>Training functions’</b> database table read from file {}'.
                format(trainfile)))
        vis_tabs.append(
            createDataTablePanel(
                df_testData, 'Test Data (x-sorted)',
                '<b>Test dataset</b> read from file {}'.format(testfile)))

        if verbose:
            print('Training Data (raw)\n' +
                  tabulate(df_trainingData, headers='keys', tablefmt='psql'))
            print('\nIdeal Data (raw)\n' + tabulate(
                df_idealData.loc[:, :'y10'], headers='keys', tablefmt='psql'))
            print('\nIdeal Data (raw)\n' +
                  tabulate(df_idealData.loc[:, 'y11':'y20'],
                           headers='keys',
                           tablefmt='psql'))
            print('\nIdeal Data (raw)\n' +
                  tabulate(df_idealData.loc[:, 'y21':'y30'],
                           headers='keys',
                           tablefmt='psql'))
            print('\nIdeal Data (raw)\n' +
                  tabulate(df_idealData.loc[:, 'y31':'y40'],
                           headers='keys',
                           tablefmt='psql'))
            print('\nIdeal Data (raw)\n' +
                  tabulate(df_idealData.loc[:, 'y41':'y50'],
                           headers='keys',
                           tablefmt='psql'))
            print('\nTest Data (raw)\n' +
                  tabulate(df_testData, headers='keys', tablefmt='psql'))

        p2f_alloc = p2fa.CPoint2FunctionAllocator()
        minLses, greatestDeviations = p2f_alloc.preselectFunctions(
            df_trainingData, df_idealData)
        greatestDeviations = greatestDeviations.rename_axis('I/T')
        vis_tabs.append(
            createDataTablePanel(
                greatestDeviations.reset_index(),
                'Greatest Deviations Map (generated)',
                'Table with <b>greatest y-deviations</b> between <b>I\
                </b>deal: <b>T</b>raining functions'))

        if verbose:
            print(
                '\nGreatest Deviations\n' +
                tabulate(greatestDeviations, headers='keys', tablefmt='psql'))
            print('Function assignments (Training: Ideal)\n{}'.format(minLses))

        df_resultTable = p2f_alloc.mapPoints2Functions(df_testData,
                                                       df_idealData, minLses,
                                                       greatestDeviations)

        vis_tabs.append(
            createMatchingPointsPanel(
                df_testData, df_resultTable, 'Matching Points (plotted)', 
                'Points from the test dataset mapped to ideal function also\
                corresponding the condition y-deviation smaller than factor\
                sqrt(2) are displayed/coloured'\
            ))
        vis_tabs.append(
            createRegressionPlotPanel(
                df_idealData, df_resultTable, 'Regressions (plotted)', 
                'Matching points from test dataset and ideal functions are\
                drawn. Also the regression lines assigned to the corresponding\
                ideal functions points'
            ))
        vis_tabs.append(
            createMappedPointsPanel(
                df_idealData, df_resultTable, df_trainingData,
                'Mapped Points (plotted)', 
                'Visualization of points from training <b>(x)</b> and ideal\
                functions <b>(+)</b> datasets. With the help of the built-in\
                zoom function, deviations(test to ideal or to regression\
                lines) can be determined or displayed'
            ))

        # write to sql database
        resultData.writeDataToDB(df_resultTable)
        df_resultData = resultData.readDataFromDB()
        vis_tabs.append(
            createDataTablePanel(
                df_resultData,
                'Result Data (generated)',
                '<b>Result</b> database table of the testdata, with mapping and y-deviation'))

        if verbose:
            print('Result Table\n' +
                  tabulate(df_resultData, headers='keys', tablefmt='psql'))

        show(Tabs(tabs=vis_tabs))

    except p2fa.LittleUsableDataError as e:
        print('Data is not usable! ')
        details = e.args[0]
        print(e)

    except p2fa.CLineTableDataShapeError as e:
        print('Data has not the expected shape! ')
        details = e.args[0]
        print(e)

    except VisualizationError as e:
        print('Data to visualize does not exist! ')
        details = e.args[0]
        print(e)

    except Exception as e:
        details = e.args[0]
        print(e)

    finally:
        print('Program finished normally!')

if __name__ == '__main__':
    runtime = timeit.Timer(lambda: main(sys.argv[1:]))
    print('Runtime: {}ms'.format(runtime.timeit(1)))
