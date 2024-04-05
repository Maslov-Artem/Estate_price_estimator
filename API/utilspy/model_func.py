import pandas as pd
import numpy as np

import geopy.distance as dis
import pickle
import joblib
import shap


from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import CatBoostEncoder


import sklearn
sklearn.set_config(transform_output="pandas")


import warnings
warnings.filterwarnings('ignore') 


from PIL import Image
import io

def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()

    image.save(imgByteArr, format=image.format)

    imgByteArr = imgByteArr.getvalue()
    return imgByteArr



import pandas as pd
import numpy as np
import geopy.distance as dis
import shap
import joblib
from PIL import Image
import io


filename_pipe = 'data/pipeline_ltl.pkl'

filename_model = 'data/model_ltl.pkl'

metro_path = 'data/metro_model.pkl'

y_mean = 15000000


def dist(c1,c2):
    return dis.geodesic(c1, c2).km

def dist_maker_df(df: pd.DataFrame, Y):
    dists = {'kremlin_dist': (55.752004, 37.617734),
             'moskva_city_dist': (55.749776, 37.536234)}
    
    for place, coord in dists.items():
        df[place] = [dist(c1, coord) for c1 in Y]


ex_filename = 'data/explainer.sav'
def image_to_byte_array(image: Image) -> bytes:
  imgByteArr = io.BytesIO()

  image.save(imgByteArr, format=image.format)

  imgByteArr = imgByteArr.getvalue()
  return imgByteArr



def Result_Maker(json):
    shap.initjs()

    metro = pd.read_pickle(metro_path)
    model = joblib.load(open(filename_model, 'rb'))
    pipe = joblib.load(open(filename_pipe, 'rb'))
    # explainer = pickle.load(open(ex_filename, 'rb'))

    X = pd.DataFrame(json, index=[0])
    X = X.rename(columns={'square':'total_area','quality':'repair_type','lat':'latitude','lan':'longitude'})

    X[['total_area','latitude','longitude']] = X[['total_area','latitude','longitude']].astype(float)
    Y = tuple(zip(X['latitude'].astype(float), X['longitude'].astype(float)))

    dist_maker_df(X,Y)

    metro['dist'] = [dist(c, Y) for c in metro.coords]
    metro_m = metro[metro['dist'] == min(metro['dist'])].iloc[0,3]

    X['metro_m'] = metro_m

    X[['latitude','longitude','total_area','moskva_city_dist','kremlin_dist']] =\
        X[['latitude','longitude','total_area','moskva_city_dist','kremlin_dist']].astype(float)
    X = X[['latitude','longitude','total_area','moskva_city_dist','kremlin_dist','repair_type','metro_m']]

    # X_SHAP = X[['latitude','longitude','total_area','moskva_city_dist','kremlin_dist','repair_type','metro_m']].round(2)
    
    X_pred = pipe.transform(X)


    # X_pred = pd.DataFrame(X_pred).rename(columns={0:'repair_type',1:'metro_m',2:'total_area',3:'moskva_city_dist',4:'kremlin_dist',5:'latitude',6:'longitude'})
    # X_pred = X_pred[['latitude','longitude','total_area','moskva_city_dist','kremlin_dist','repair_type','metro_m']]

    result = int(np.round(model.predict(X_pred),-5)[0])

    # shap_v = explainer.shap_values(X_pred)

    # image = shap.force_plot(y_mean, shap_v, X_SHAP, matplotlib=True,text_rotation=30, contribution_threshold=0.1, show=False)
    # image.savefig('s.png')
    


    # bytes_ = image_to_byte_array(Image.open('s.png'))

    return result, metro_m