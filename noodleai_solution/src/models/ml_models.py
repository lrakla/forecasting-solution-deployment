import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class XGB:
    def __init__(self, params):
        self.model = XGBRegressor(**params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        # dtest = xgb.DMatrix(data=X_test, enable_categorical=True)
        return self.model.predict(X_test)
   
class RandomForest:
    def __init__(self, params):
        self.params = params

    def fit(self, X_train, y_train):
        model = RandomForestRegressor(self.params)
        model.fit(X_train, y_train)
        return model
    
    def predict(self, X_test):
        return np.exp(self.model.predict(self.X_test))
    


