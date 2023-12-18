from typing import List, Optional, Tuple, Union
from src.visualizations.visualize import plotForecasts
import numpy as np

class evaluate:
    def __init__(self,df, actual_col,prediction_col,  metric:str, to_plot = False):
        '''
        Args:
            :model: Time-series model to be used for forecast
            :actuals: actual values
            :metric: metric to be used for evaluation

        '''
        self.df  = df
        self.metric = metric
        self.actual_col = actual_col
        self.prediction_col = prediction_col
        self.to_plot = to_plot
    
    def evaluate(self):
        '''
        
        Return

            : metric value
        '''
        self.actuals = self.df[self.actual_col]
        self.forecast = self.df[self.prediction_col]
        if self.to_plot:
            self.plot()

        if self.metric == 'mape':
            return self._mape(self.actuals, self.forecast)
        elif self.metric == 'smape':
            return self._smape(self.actuals, self.forecast)
        elif self.metric == 'wmape':
            return self._weighted_mape(self.actuals, self.forecast)
        elif self.metric == 'all':
            return {'mape': self._mape(self.actuals, self.forecast),
                    'smape': self._smape(self.actuals, self.forecast),
                    'wmape': self._weighted_mape(self.actuals, self.forecast)}
        else:
            raise Exception('Metric not supported')

    def _weighted_mape(self,y_true, y_pred):
        ''' 
        Compute weighted mean absolute percentage error (WMAPE)
        Args:
            :y_true: array containing actual values
            :y_pred: array containing predicted values
        Return
            : WMAPE
        '''
        ape = np.abs((y_true - y_pred))
        wmape_value = np.sum(ape) / np.sum(np.abs(y_true))
        return wmape_value

    def _mape(self,y_true, y_pred):
        ''' 
        Compute mean absolute percentage error (MAPE)
        Args:
            :y_true: array containing actual values
            :y_pred: array containing predicted values
        Return
            : MAPE
        '''
        ape = np.abs((y_true - y_pred)/y_true)
        mape_value = np.mean(ape)
        return mape_value

    def _smape(self,y_true, y_pred):
        ''' 
        Compute symmetric mean absolute percentage error (SMAPE)
        Args:
            :y_true: array containing actual values
            :y_pred: array containing predicted values
        Return
            : SMAPE
        '''
        ape = np.abs((y_true - y_pred))
        smape_value = np.mean(ape / (np.abs(y_true) + np.abs(y_pred)))
        return smape_value
        
    def plot(self):
        '''
        Plot actuals vs forecast
        '''
        plotForecasts().plot(self.df,self.actual_col ,self.prediction_col)
