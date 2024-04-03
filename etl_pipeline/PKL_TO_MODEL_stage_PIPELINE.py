import pandas as pd
import numpy as np
import datetime


from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin


import sklearn
sklearn.set_config(transform_output="pandas")
import warnings
warnings.filterwarnings('ignore')






####################################################################################################################

class Rooms(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()

        def rooms(area):
            if area <= 40:
                return 1
            # elif area <= 80:
            return 2
            # elif area <= 120:
            #     return 3
            # return 4
        
        X_copy['room_count'] = [rooms(area) if room in [0,'0','False',False,'Нет данных'] or np.isnan(room) else room for area, room in X_copy[['total_area','room_count']].itertuples(index=False)]

        height = X_copy.loc[X_copy.ceiling_height != 'Нет данных'].ceiling_height.astype(float).median()

        def ceil(x):
            if x == 'Нет данных':
                return height
            x = float(x)

            if x <= 1.5:
                return height
            return x

        X_copy['ceiling_height'] = X_copy['ceiling_height'].map(ceil)

        metro_year = pd.read_pickle('metro_year.pkl')

        def f(year):
            if year == 'Нет данных':
                return np.NaN
            else:
                return int(year)
                
        X_copy['build_year'] = X_copy['build_year'].map(f)

        def ff(x):
            return metro_year.loc[metro_year.index == x].iloc[0].item()
        
        
        X_copy['build_year'] = [ff(station) if year <= 1500 or np.isnan(year) else year for year, station in X_copy[['build_year','metro_m']].itertuples(index=False)]
        return X_copy


############################################################################################################################################################################################################


first_stage = ['total_area','room_count','ceiling_height','build_year','metro_m']

first_step = ColumnTransformer(
    transformers=[
        ('rooms', Rooms(), first_stage),
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')

preprocessor_2 = Pipeline(
    [
        ('some_ltl_transformations', first_step),
    ]
)

######################################################################################################################################################################################

#1 считываем данные
data = pd.read_pickle('data_comb.pkl')

#2 preprocessor

dataN = preprocessor_2.fit_transform(data)


#3 ремонт и материалы дома




#4 catboost




#5 экспорт данных 