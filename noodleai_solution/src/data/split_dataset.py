import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml
import os

with open(os.path.dirname(__file__) + '/../CONFIG.yaml', 'r') as file:
    yaml_content = yaml.safe_load(file)

# Access global variables from CONFIG.yaml
cat_vars = yaml_content['cat_vars']
num_vars = yaml_content['num_vars']
features = yaml_content['features']
lags = yaml_content['lags']
transform_target = yaml_content['transform_target']

actual_col = yaml_content['actual_col']

if transform_target:
    target_col = yaml_content['target_col']
else:
    target_col = actual_col
    
train_split_date = pd.to_datetime(yaml_content['train_split_date'])
lag_cols = []
if lags:
    for lag in range(1,lags+1):
        lag_cols.append('lag_{}'.format(lag))


class splitDataset:
    def __init__(self, df, scale = True):
        '''
        Args:
            :df: dataframe
            :scale: boolean to indicate whether to scale the data or not
        '''
        self.df = df
        self.scale = scale

    def run(self):
        '''
        Split the data into train, validation and test sets
        Returns:
            :X_train: train features
            :y_train: train target
           
            :X_test: test features
            :y_test: test target
            :train_date: train dates
            :test_date: test dates
        ''' 
        df_train = self.df[self.df['Date'] <= train_split_date]
        df_train = self._create_lag_features(df_train)
        df_test = self.df[self.df['Date'] > train_split_date]


        X_train = df_train[features + lag_cols]
        y_train = df_train[target_col]
      
        actuals = df_test[actual_col]
        X_test = df_test[features]
        y_test = df_test[target_col]

        train_date = df_train['Date']
        test_date = df_test['Date']

        if self.scale:
            scaler = MinMaxScaler()
            X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
            X_test[num_vars] = scaler.transform(X_test[num_vars])
        return X_train, y_train, X_test, y_test, actuals, train_date, test_date
    
    def _create_lag_features(self,df):
        '''
        Create lag features
        '''
     
        df = df.sort_values(by=['Store', 'Dept', 'Week'])
        for lag in range(1,lags+1):
            df['lag_{}'.format(lag)] = df.groupby(['Store', 'Dept'])['Weekly_Sales_transformed'].shift(lag)

        # if there are any null values, fill them with weekly_sales_transformed
        for lag in range(1,lags+1):
            df['lag_{}'.format(lag)].fillna(df['Weekly_Sales_transformed'], inplace = True)
        return df
