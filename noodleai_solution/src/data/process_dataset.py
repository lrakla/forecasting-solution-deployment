# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

class processDataset:
    '''
    Preprocess the dataset
    '''
    def __init__(self, input_filepath, output_filepath):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

    def run(self):
        # load data
        data = pd.read_csv(self.input_filepath)
        # preprocess data
        data = self._preprocess(data)
        # save data
        self._save_data(data)
        return data
    
    def _preprocess(self, data):
        '''
        CHnage types, impute missing values and transform skewed target variable
        '''
        # convert date column to datetime
        data['Date'] = pd.to_datetime(data['Date'], format = '%Y-%m-%d')
        data['Store'] = data['Store'].astype('category')
        data['Dept'] = data['Dept'].astype('category')
        data['ts_key'] = data['ts_key'].astype('category')
        
        # Impute missing values
        for i in range(1,6):
            data['MarkDown{}'.format(i)].fillna(0, inplace = True)
        data['Unemployment'].fillna(data['Unemployment'].mean(), inplace = True)
        # change negative wekely sales to one so log transform works
        data['Weekly_Sales'] = [1 if x <= 0 else x for x in data['Weekly_Sales']]
        data['Weekly_Sales_transformed'] = data['Weekly_Sales'].apply(lambda x: np.log(x))
        return data
    
    def _save_data(self, data):
        data.to_csv(self.output_filepath, index = False)
    

        

# Code to change format of data to a json for easier analysis
# data_dict = {}
# # Iterate over each row in the DataFrame
# for index, row in data.iterrows():
#     # Get the date, store, and department
#     date = str(row['Date'])
#     store = row['Store']
#     dept = row['Dept']

#     # If the date is not in the dictionary, add it
#     if date not in data_dict:
#         data_dict[date] = {}

#     # If the store is not in the date's dictionary, add it
#     if store not in data_dict[date]:
#         data_dict[date][store] = {'Type': row['Type'], 'Size': row['Size'], 'Temperature': row['Temperature'], 'Fuel_Price': row['Fuel_Price'], 'MarkDown1': row['MarkDown1'], 'MarkDown2': row['MarkDown2'], 'MarkDown3': row['MarkDown3'], 'MarkDown4': row['MarkDown4'], 'MarkDown5': row['MarkDown5'], 'CPI': row['CPI'], 'Unemployment': row['Unemployment'], 'IsHoliday': row['IsHoliday'], 'ts_key': row['ts_key'], 'Depts': {}, 'Total_Weekly_Sales': 0}

#     # Add the department and its weekly sales to the store's dictionary
#     data_dict[date][store]['Depts'][dept] = row['Weekly_Sales']
#     data_dict[date][store]['Total_Weekly_Sales'] += row['Weekly_Sales']

# # save the data_dict as json
# import json
# with open('../data/processed/data_dict.json', 'w') as fp:
#     json.dump(data_dict, fp)