Time-Series-Sales-Forecasting
==============================

To get started - 
1. Clone this repository
2. Run the following command
'''python
docker build -t noodleai_solution .
docker run -d noodleai_solution
'''


import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import sklearn
import xgboost as xgb