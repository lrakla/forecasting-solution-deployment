import yaml
import os
from src.models.ml_models import XGB, RandomForest
from src.evaluate import *
import pandas as pd

with open(os.path.dirname(__file__) + '/models_config.yaml', 'r') as file:
      models_config = yaml.safe_load(file)

with open(os.path.dirname(__file__) + '/CONFIG.yaml', 'r') as file:
    yaml_content = yaml.safe_load(file)
    filepaths = yaml_content['filepaths']

results_filepath = filepaths['results_filepath']
transform_target = yaml_content['transform_target']

def train_pipeline(X_train, y_train, X_test, y_test,actuals,test_date):
    models = {'XGBoost' : models_config['params_xgb']
            #   ,'RandomForest' : yaml_content['params_rf']
              }
    for model,params in models.items():
        if model == 'XGBoost':
            model = XGB(params)
        elif model == 'RandomForest':
            model = RandomForest(params)
        
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        if transform_target:
            test_preds = np.exp(test_preds)
        merged_test = merge_frames(X_test, actuals, test_preds, test_date)
        print('Test results for {} model : '.format(model), test(merged_test, 'Actual', 'Forecast'))
        merged_test.to_csv(results_filepath, index = False)
        return merged_test


#helper
def merge_frames(df, actuals, preds, date):
    df['Actual'] = actuals
    df['Forecast'] = preds
    df['Date'] = date
    return df
#helper
def test(df, actual_col, prediction_col):
    eval = evaluate(df, actual_col, prediction_col, metric = 'all', to_plot =  True)
    return eval.evaluate()