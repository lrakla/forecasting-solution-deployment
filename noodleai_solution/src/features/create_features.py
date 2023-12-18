from sklearn.preprocessing import LabelEncoder
import pandas as pd

class createFeatures:
    def __init__(self, data, output_filepath):
        self.data = data
        self.output_filepath = output_filepath
    def run(self):
        self._create_time_features()
        self._encode_features()
        self._save_data(self.data)
        return self.data
    
    def _create_time_features(self):
        # create a column for year
        self.data['Year'] = self.data['Date'].dt.year
        # create a column for month
        self.data['Month'] = self.data['Date'].dt.month
        # create a column for week of year
        self.data['Week'] = self.data['Date'].dt.isocalendar().week
    
    def _encode_features(self):
        self.data['IsHoliday'] = [1 if x  else 0 for x in self.data['IsHoliday']]
        self.data['Store'] = LabelEncoder().fit_transform(self.data['Store'])
        self.data['Dept'] = LabelEncoder().fit_transform(self.data['Dept'])
        self.data['Type'] = LabelEncoder().fit_transform(self.data['Type'])

    def _save_data(self, data):
        data.to_csv(self.output_filepath, index = False)