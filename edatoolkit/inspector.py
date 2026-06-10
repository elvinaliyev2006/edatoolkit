import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class Inspector:

    def __init__(self):
        self.line = '─' * 170
    

    def get_columns_types(self,dataframe,car_th=20,cat_th=10):
        """
        Sütunları tip əsasında 4 qrupa ayırır.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Analiz ediləcək dataset.
        car_th : int, optional
            Kateqorial sütun üçün yüksək kardinallik həddi (default: 20).
        cat_th : int, optional
            Rəqəmsal sütunu kateqorial saymaq üçün unikal dəyər həddi (default: 10).

        Returns
        -------
        tuple
            (cat_cols, num_cols, num_but_cat, cat_but_car)
        """
        cat_cols = [col for col in dataframe.columns if
                    str(dataframe[col].dtype) in ['object', 'bool', 'category','str', 'string']]
        num_but_cat = [col for col in dataframe.columns if
                       pd.api.types.is_numeric_dtype(dataframe[col]) and dataframe[
                           col].nunique() < cat_th]
        cat_but_car = [col for col in dataframe.columns if
                       str(dataframe[col].dtype) in ['object', 'category'] and dataframe[
                           col].nunique() > car_th]
        c_c = [col for col in num_but_cat if col not in cat_cols ]
        cat_cols = cat_cols + c_c
        cat_cols = [col for col in cat_cols if col not in cat_but_car]
        num_cols = [col for col in dataframe.columns if
                    pd.api.types.is_numeric_dtype(dataframe[col]) and col not in num_but_cat]
        return cat_cols, num_cols, num_but_cat, cat_but_car

    def check_dataframe(self,dataframe, n=5):

        """
        Prints a general overview of the dataframe including head, sample, tail,
        shape, info, missing values, and duplicate counts.

        Parameters
        ----------
        n : int, optional
            Number of rows to display for head/sample/tail (default: 5).
        """

        print(f'\n{self.line}')
        print(' Head '.center(170))
        print(self.line)
        print(dataframe.head(n))
        print(f'\n{self.line}')
        print(' Sample '.center(170))
        print(self.line)
        print(dataframe.sample(n))
        print(f'\n{self.line}')
        print(' Tail '.center(170))
        print(self.line)
        print(dataframe.tail(n))
        print(f'\n{self.line}')
        print(' Shape '.center(170))
        print(self.line)
        print('Rows: ', dataframe.shape[0])
        print('Columns: ', dataframe.shape[1])
        print(f'\n{self.line}')
        print(' Info '.center(170))
        print(self.line)
        print(dataframe.info())
        print(f'\n{self.line}')
        print(' NA '.center(170))
        print(self.line)
        print(dataframe.isnull().mean())
        print(f'\n{self.line}')
        print(' Duplicate Values '.center(170))
        print(self.line)
        print('Count: ', dataframe.duplicated().sum())
        print('Ratio: ', (dataframe.duplicated().sum()) / (dataframe.shape[0]))    