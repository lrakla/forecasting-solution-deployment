import yaml
import os
from src.models.ml_models import XGB, RandomForest
from src.evaluate import *

with open(os.path.dirname(__file__) + '/models_config.yaml', 'r') as file:
    yaml_content = yaml.safe_load(file)


def experiments(X_train, y_train, X_test, y_test, X_val, y_val, train_date, val_date, test_date):
    models = {'XGBoost' : yaml_content['params_xgb']
            #   ,'RandomForest' : yaml_content['params_rf']
              }
    for model,params in models.items():
        if model == 'XGBoost':
            model = XGB(params)
        elif model == 'RandomForest':
            model = RandomForest(params)
        
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        merged_val = merge_frames(X_val, y_val, val_preds, val_date)
        merged_test = merge_frames(X_test, y_test, test_preds, test_date)
        print('Validation results for {} model : '.format(model),test(merged_val, 'Actual', 'Forecast'))
        print('Test results for {} model : '.format(model), test(merged_test, 'Actual', 'Forecast'))
        return merged_test


#helper
def merge_frames(features, target, preds, date):
    features['Actual'] = target
    features['Forecast'] = preds
    features['Date'] = date
    return features
#helper
def test(df, actual_col, prediction_col):
    eval = evaluate(df, actual_col, prediction_col, metric = 'all', to_plot =  True)
    return eval.evaluate()