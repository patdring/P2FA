import pandas as pd
import sqlalchemy
import pymysql
import sqlalchemy as db
import csv

class IdealData:
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

    def __init__(self, csv_file, table_name, db_user='root', db_password='Tr5jeb6X!', db_address='127.0.0.1', db_name='test'):
        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
            name : str
                first name of the person
            surname : str
                family name of the person
            age : int
                age of the person
        """

        self._table_name = table_name
        # get engine object using pymysql driver for mysql
        self._engine = db.create_engine('mysql+pymysql://{}:{}@{}/{}'.format(db_user, db_password, db_address, db_name))
        # get metadata object
        self._meta_data = db.MetaData()
        # get connection object
        self._connection = self._engine.connect()
        self._readCSV(csv_file)
        
    def _readCSV(self, csv_file, csv_del = ','):
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

        self._data = pd.read_csv(csv_file, delimiter = csv_del)
        self._data = self._data.set_index('x')
        self._data.to_sql(self._table_name, self._engine, if_exists='replace')

    def readDataFromDB(self):
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

        return pd.read_sql_table(self._table_name, self._connection)

class TrainingData(IdealData):
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

    def __init__(self, csv_file, table_name, db_user='root', db_password='Tr5jeb6X!', db_address='127.0.0.1', db_name='test'):
        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
            name : str
                first name of the person
            surname : str
                family name of the person
            age : int
                age of the person
        """

        super().__init__(csv_file, table_name, db_user, db_password, db_address, db_name)

    def _readCSV(self, csv_file, csv_del = ','):
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

        df_old = pd.DataFrame(None, columns = ['x'])   
        for i in range(0,len(csv_file)):
            df1 = pd.read_csv(csv_file[i],delimiter = csv_del)
            df1 = df1.rename(columns={'y': 'y{}'.format(i+1)})
            df1 = pd.merge(df_old, df1, how='outer')
            df_old = df1
            
        self._data = df_old.set_index('x')
        self._data.to_sql(self._table_name, self._engine, if_exists='replace')
                  
class TestData(IdealData):
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

    def __init__(self, csv_file, table_name, db_user='root', db_password='Tr5jeb6X!', db_address='127.0.0.1', db_name='test'):
        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
            name : str
                first name of the person
            surname : str
                family name of the person
            age : int
                age of the person
        """

        super().__init__(csv_file, table_name, db_user, db_password, db_address, db_name)

    def _readCSV(self, csv_file, csv_del = ','):     
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

        with open(csv_file) as f:
            reader = csv.DictReader(f, delimiter = csv_del)
            column_names = reader.fieldnames
            self._data = pd.DataFrame(None, columns=column_names)
            for row in reader:
                self._data.loc[len(self._data)] = row

        self._data = pd.read_csv(csv_file, delimiter = csv_del)
        self._data = self._data.set_index('x')
        self._data.to_sql(self._table_name, self._engine, if_exists='replace')

class ResultData(IdealData):
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

    def __init__(self, table_name, csv_file=None, db_user='root', db_password='Tr5jeb6X!', db_address='127.0.0.1', db_name='test'):
        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
            name : str
                first name of the person
            surname : str
                family name of the person
            age : int
                age of the person
        """

        super().__init__(csv_file, table_name, db_user, db_password, db_address, db_name)
        self._data = None

    def _readCSV(self, csv_file, csv_del = ','):
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

        pass

    def writeDataToDB(self, data):
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

        self._data = data
        self._data = self._data.set_index('x')
        self._data.to_sql(self._table_name, self._engine, if_exists='replace')
