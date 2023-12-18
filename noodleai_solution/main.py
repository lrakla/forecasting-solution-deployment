
import logging
from src.data.process_dataset import processDataset
from src.data.split_dataset import splitDataset
from src.features.create_features import createFeatures
from src.train import train_pipeline
import yaml
import warnings
warnings.filterwarnings("ignore")
from src.visualizations.visualize import plotForecasts
import argparse


def getForecast_for_store_dept(test_df,store = None, dept = None):
    '''
    Function to get forecast for a given store and department
    '''
    assert test_df[test_df['Store'] == store].shape[0] > 0, "Store not found in test set"
    if store and dept:
        assert test_df[(test_df['Store'] == store) & (test_df['Dept'] == dept)].shape[0] > 0, "This store and department combination not found in test set"
    if store and dept:
        forecasts = test_df[(test_df['Store'] == store) & (test_df['Dept'] == dept)][['Date','Forecast']]
    elif store and not dept:
        forecasts = test_df[test_df['Store'] == store][['Date','Forecast']]
    else:
        forecasts = test_df[['Date','Forecast']]
    plotForecasts().plot(test_df, 'Actual','Forecast',store = store, dept  = dept)
    return forecasts

def main(args):
    '''
    Main function to run the pipeline

    '''
    store = args.store
    dept = args.dept

    with open("src/CONFIG.yaml", 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        filepaths = config['filepaths']
    input_filepath = filepaths['input_filepath']
    processed_filepath = filepaths['processed_filepath']
    encoded_filepath = filepaths['encoded_filepath']

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='log.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG) 


    logger.info('Starting to build dataset...')
    process_dataset = processDataset(input_filepath, processed_filepath)
    data = process_dataset.run()
    create_features = createFeatures(data,encoded_filepath)
    data = create_features.run()
    split_dataset = splitDataset(data,scale = True)
    X_train, y_train, X_test, y_test, actuals, train_date, test_date = split_dataset.run()
    logger.info('Finished building dataset...')

    # train models
    logger.info('Starting to run experiments...')
    test_df = train_pipeline(X_train, y_train, X_test, y_test, actuals, test_date)
    logger.info('Finished experiments...')
    # visualize
    print(test_df.head())

    if store is not None or dept is not None:
        forecasts = getForecast_for_store_dept(test_df,store,dept)
        print(forecasts.head())
        
    return test_df

if __name__ == '__main__':

    argparse = argparse.ArgumentParser()
    argparse.add_argument('--store', type=int, default=None, help='Store number')
    argparse.add_argument('--dept', type=int, default=None, help='Department number')

    args = argparse.parse_args()
    main(args)

    

