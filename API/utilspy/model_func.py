import pandas as pd
import numpy as np

import geopy.distance as dis
import pickle
import joblib



from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import CatBoostEncoder


import sklearn
sklearn.set_config(transform_output="pandas")
import warnings
warnings.filterwarnings('ignore')


class Nummer(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()
        return X_copy.astype(float)



def dist(c1,c2):
    return dis.geodesic(c1, c2).km

def dist_maker_df(df: pd.DataFrame, Y):
    dists = {'kremlin_dist': (55.752004, 37.617734),
             'moskva_city_dist': (55.749776, 37.536234)}
    
    for place, coord in dists.items():
        df[place] = [dist(c1, coord) for c1 in Y]


def Result_Maker(json):

    metro = pd.read_pickle('data/metro_model.pkl')
    model = joblib.load(open('data/ltl_model.pkl', 'rb'))

    X = pd.DataFrame(json, index=[0])
    X = X.rename(columns={'square':'total_area','quality':'repair_type','lat':'latitude','lan':'longitude'})

    X[['total_area','latitude','longitude']] = X[['total_area','latitude','longitude']].astype(float)
    Y = tuple(zip(X['latitude'].astype(float), X['longitude'].astype(float)))

    dist_maker_df(X,Y)

    metro['dist'] = [dist(c, Y) for c in metro.coords]
    metro_m = metro[metro['dist'] == min(metro['dist'])].iloc[0,3]
    X['metro_m'] = metro_m

    X = X[['latitude','longitude','total_area','moskva_city_dist','kremlin_dist','repair_type','metro_m']]
    
    pred = int(np.round(np.exp(model.predict(X)),-5))
    return pred, metro_m