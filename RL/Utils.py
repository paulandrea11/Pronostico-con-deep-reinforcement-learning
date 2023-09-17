import pandas as pd
import numpy as np
import os

class Utils:
    def data_split(self, df, start, end):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(start <= df.datadate) & (df.datadate <= end)]
        data = data.sort_values(['datadate','tic'], ignore_index=True)
        #data  = data[final_columns]
        data.index = data.datadate.factorize()[0]
        return data

    def get_validation_sharpe(self, iteration, PATH_RESULTS = '/content/results'):
        ###Calculate Sharpe ratio based on validation /content/results###
        df_total_value = pd.read_csv(PATH_RESULTS+'/account_value_validation_{}.csv'.format(iteration), index_col=0)
        df_total_value.columns = ['account_value_train']
        df_total_value['daily_return'] = df_total_value.pct_change(1)
        sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
        return sharpe

