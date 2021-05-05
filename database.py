import pandas as pd
import sqlalchemy
import pymysql
import sqlalchemy as db
import csv

class IdealData:
    def __init__(self, csv_file, table_name, db_user='root', db_password='Tr5jeb6X!', db_address='127.0.0.1', db_name='test'):
        self._table_name = table_name
        # get engine object using pymysql driver for mysql
        self._engine = db.create_engine('mysql+pymysql://{}:{}@{}/{}'.format(db_user, db_password, db_address, db_name))
        # get metadata object
        self._meta_data = db.MetaData()
        # get connection object
        self._connection = self._engine.connect()
        self._readCSV(csv_file)
        
    def _readCSV(self, csv_file):
        self._data = pd.read_csv(csv_file, delimiter = ';')
        self._data = self._data.set_index('X')
        self._data.to_sql(self._table_name, self._engine, if_exists='replace')

    def readDataFromDB(self):
        return pd.read_sql_table(self._table_name, self._connection)

class TrainingData(IdealData):
    def __init__(self, csv_file, table_name, db_user='root', db_password='Tr5jeb6X!', db_address='127.0.0.1', db_name='test'):
        super().__init__(csv_file, table_name, db_user, db_password, db_address, db_name)

    def _readCSV(self, csv_file):

        df_old = pd.DataFrame(None, columns = ['X'])   
        for i in range(0,len(csv_file)):
            df1 = pd.read_csv(csv_file[i],delimiter = ';')
            df1 = df1.rename(columns={'Y': 'Y{}'.format(i+1)})
            df1 = pd.merge(df_old, df1, how='outer')
            df_old = df1
            
        self._data = df_old.set_index('X')
        self._data.to_sql(self._table_name, self._engine, if_exists='replace')
                  
class TestData(IdealData):
    def __init__(self, csv_file, table_name, db_user='root', db_password='Tr5jeb6X!', db_address='127.0.0.1', db_name='test'):
        super().__init__(csv_file, table_name, db_user, db_password, db_address, db_name)

    def _readCSV(self, csv_file):     
        with open(csv_file) as f:
            reader = csv.DictReader(f, delimiter=";")
            column_names = reader.fieldnames
            self._data = pd.DataFrame(None, columns=column_names)
            for row in reader:
                self._data.loc[len(self._data)] = row

        self._data = pd.read_csv(csv_file, delimiter = ';')
        self._data = self._data.set_index('X')
        self._data.to_sql(self._table_name, self._engine, if_exists='replace')

class ResultData(IdealData):
    def __init__(self, table_name, csv_file=None, db_user='root', db_password='Tr5jeb6X!', db_address='127.0.0.1', db_name='test'):
        super().__init__(csv_file, table_name, db_user, db_password, db_address, db_name)
        self._data = None

    def _readCSV(self, csv_file):
        pass

    def writeDataToDB(self, data):
        self._data = data
        self._data.to_sql(self._table_name, self._engine, if_exists='replace')

#TODO delimier option