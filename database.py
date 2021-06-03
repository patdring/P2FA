import pandas as pd
import sqlalchemy
import pymysql
import sqlalchemy as db
import csv

class CBasicTableData:
    '''
    A base class which provides a serialization layer from a csv file to a 
    mysql database. The access is configured at initialization (per 
    constructor)

    Only public attributes and methods are mentioned!

    Attributes:
            -

    Methods:
            readDataFromDB(): 
                    returns db table as pandas.DataFrame
    '''

    def __init__(self,
                 csv_file,
                 table_name,
                 db_user='root',
                 db_password='Tr5jeb6X!',
                 db_address='127.0.0.1',
                 db_name='test'):
        '''
        Constructs all the necessary attributes for the database 
        seriallization object. 

            Parameters:
                table_name (str): mysql table name 
                csv_file (str): CSV filename
                db_user (str): mysql user name
                db_password (str): mysql user password
                db_address (str): mysql server ip address
                db_name (str): mysql database name
            Returns:
                -
        '''

        self._table_name = table_name
        # get engine object using pymysql driver for mysql
        self._engine = db.create_engine('mysql+pymysql://{}:{}@{}/{}'.format(
            db_user, db_password, db_address, db_name))
        # get metadata object
        self._meta_data = db.MetaData()
        # get connection object
        self._connection = self._engine.connect()
        self._readCSV(csv_file)

    def _readCSV(self, csv_file, csv_del=','):
        '''
        Reads a CSV file and stores it in a database and table set by 
        constructor. Existing tables will be overwritten!

            Parameters:
                csv_file (str): Filename of the CSV file 
                csv_del (str): CSV delimiter character
            Returns:
                -
        '''

        self._data = pd.read_csv(csv_file, delimiter=csv_del)
        self._data = self._data.set_index('x')
        self._data.to_sql(self._table_name, self._engine, if_exists='replace')

    def readDataFromDB(self):
        '''
         Reads Data (a Table) from mysql database
         
            Parameters:
                csv_file (list): List of CSV filenames
                csv_del (str): CSV delimiter character
            Returns:
                Deseriallized db table (pandas.DataFrame)
        '''

        return pd.read_sql_table(self._table_name, self._connection)


class CMultipleTableData(CBasicTableData):
    '''
    A derived class which provides a serialization layer from csv files to a 
    mysql database. The access is configured at initialization (per 
    constructor)

    Only public attributes and methods are mentioned!

    Attributes:
            -

    Methods:
            -
    '''

    def __init__(self,
                 csv_file,
                 table_name,
                 db_user='root',
                 db_password='Tr5jeb6X!',
                 db_address='127.0.0.1',
                 db_name='test'):
        '''
        Constructs all the necessary attributes for the database 
        seriallization object. 

            Parameters:
                table_name (str): mysql table name 
                csv_file (str): CSV filename
                db_user (str): mysql user name
                db_password (str): mysql user password
                db_address (str): mysql server ip address
                db_name (str): mysql database name
            Returns:
                -
        '''

        super().__init__(csv_file, table_name, db_user, db_password,
                         db_address, db_name)

    def _readCSV(self, csv_file, csv_del=','):
        '''
         Reads multiple CSV files (a list), merges them and stores 
         it in a database and table set by constructor.
         Existing tables will be overwritten!

            Parameters:
                csv_file (list): List of CSV filenames
                csv_del (str): CSV delimiter character
            Returns:
                -
        '''

        df_old = pd.DataFrame(None, columns=['x'])
        for i in range(0, len(csv_file)):
            df_new = pd.read_csv(csv_file[i], delimiter=csv_del)
            df_new = df_new.rename(columns={'y': 'y{}'.format(i + 1)})
            df_new = pd.merge(df_old, df_new, how='outer')
            df_old = df_new

        self._data = df_old.set_index('x')
        self._data.to_sql(self._table_name, self._engine, if_exists='replace')


class CLineTableData(CBasicTableData):
    '''
    A base class which provides a serialization layer from csv file to a 
    mysql database. The access is configured at initialization (per 
    constructor)

    Only public attributes and methods are mentioned!

    Attributes:
            -

    Methods:
            -
    '''

    def __init__(self,
                 csv_file,
                 table_name,
                 db_user='root',
                 db_password='Tr5jeb6X!',
                 db_address='127.0.0.1',
                 db_name='test'):
        '''
        Constructs all the necessary attributes for the database 
        seriallization object. 

            Parameters:
                table_name (str): mysql table name 
                csv_file (str): CSV filename
                db_user (str): mysql user name
                db_password (str): mysql user password
                db_address (str): mysql server ip address
                db_name (str): mysql database name
            Returns:
                -
        '''

        super().__init__(csv_file, table_name, db_user, db_password,
                         db_address, db_name)

    def _readCSV(self, csv_file, csv_del=','):
        '''
        Reads a CSV file line by line and stores it in a database and table 
        set by constructor. Existing tables will be overwritten!

            Parameters:
                csv_file (str): Filename of the CSV file 
                csv_del (str): CSV delimiter character
            Returns:
                -
        '''

        with open(csv_file) as f:
            reader = csv.DictReader(f, delimiter=csv_del)
            column_names = reader.fieldnames
            self._data = pd.DataFrame(None, columns=column_names)
            for row in reader:
                self._data.loc[len(self._data)] = row

        self._data = pd.read_csv(csv_file, delimiter=csv_del)
        self._data = self._data.set_index('x')
        self._data.to_sql(self._table_name, self._engine, if_exists='replace')


class ResultData(CBasicTableData):
    '''
    A derived class which provides a deserialization layer from a 
    mysql database. The access is configured at initialization (per 
    constructor)

    Only public attributes and methods are mentioned!

    Attributes:
            -

    Methods:
            writeDataToDB(data): 
                    writes a pandas.DataFrame to a db table 
    '''

    def __init__(self,
                 table_name,
                 csv_file=None,
                 db_user='root',
                 db_password='Tr5jeb6X!',
                 db_address='127.0.0.1',
                 db_name='test'):
        '''
        Constructs all the necessary attributes for the database 
        deseriallization object. 

            Parameters:
                table_name (str): mysql table name 
                csv_file (str): CSV filename
                db_user (str): mysql user name
                db_password (str): mysql user password
                db_address (str): mysql server ip address
                db_name (str): mysql database name
            Returns:
                -
        '''

        super().__init__(csv_file, table_name, db_user, db_password,
                         db_address, db_name)
        self._data = None

    def _readCSV(self, csv_file, csv_del=','):
        '''
        Dummy of this method as implemtation for design/inheritance
        '''

        pass

    def writeDataToDB(self, data):
        '''
        Writes/Replaces data to a database and table configured by constructor

            Parameters:
                data (pandas.DataFrame): Data to be stored in db
            Returns:
                -
        '''

        self._data = data
        self._data = self._data.set_index('x')
        self._data.to_sql(self._table_name, self._engine, if_exists='replace')
