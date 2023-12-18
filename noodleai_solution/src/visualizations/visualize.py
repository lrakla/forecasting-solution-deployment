import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
class plotForecasts:
    def __init__(self):
        pass

    def plot(self,df, actual_col, prediction_col,store = None, dept = None):
        if store is not None and dept is None:
            df = df[df['Store'] == store]
        elif dept is not None and store is not None:
            df = df[(df['Store'] == store) & (df['Dept'] == dept)]
        plt.figure(figsize=(20, 7))
        sns.lineplot(x = 'Date', y=actual_col, data=df, marker='o', label='Actuals', estimator=np.sum, ci=None)
        sns.lineplot(x = 'Date', y=prediction_col, data=df, marker='o', label='Forecasts', estimator=np.sum, ci='sd')
        plt.xlabel('Time Period')
        plt.ylabel('Weekly Sales')
        if store and dept:
            plt.title('Actuals vs. Forecasts for store {} and department {}'.format(store, dept))
        else:
            plt.title('Actuals vs. Forecasts')
        plt.grid()
        plt.legend()
        plt.show()
        