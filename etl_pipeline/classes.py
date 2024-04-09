import geopy.distance as dis
import pandas as pd
import string
import numpy as np
import datetime
from tqdm import tqdm
import requests

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

import sklearn
sklearn.set_config(transform_output="pandas")
import warnings
warnings.filterwarnings('ignore')


# FUNCTIONS

def strr(x):
    x = x.lower()
    x = x.replace('ё','е')
    
    for i in string.punctuation + 'iI':
        x = x.replace(i, '')
    
    x = x.replace('зеленоград — ','')
    x = x.replace('аэропорт внуково','внуково')
    x = x.replace('бульвар генерала карбышева','народное ополчение')
    x = x.replace('троицк','ольховая')
    x = x.replace('ватутинки','ольховая')
    x = x.replace('десна','ольховая')
    x = x.replace('кедровая','ольховая')
    x = x.replace('москватоварная','москва товарная')
    x = x.replace('остров мечты','народное ополчение')
    x = x.replace('карамышевская','народное ополчение')
    x = x.replace('матвеевская','аминьевская')
    x = x.replace('летово','ольховая')
    x = x.replace('москвасити','деловой центр')
    x = x.replace('звенигородская','кунцевская')
    x = x.replace('бачуринская','ольховая')
    x = x.replace('университет дружбы народов','югозападная')
    x = x.replace('потапово','бунинская аллея')
    x = x.replace('каспийская','царицыно')
    x = x.replace('гольяново','щелковская')
    x = x.replace('кавказский бульвар','царицыно')
    x = x.replace('тютчевская','коньково')
    x = x.strip()

    
    return x


try:
    url = 'https://www.cbr-xml-daily.ru/latest.js'
    res = requests.get(url)
    usd = 1/res.json()['rates']['USD']
    eur = 1/res.json()['rates']['EUR']
except:
    usd = 92
    eur = 100

def dist(c1,c2):
    try:
        return dis.geodesic(c1, c2).km
    except:
        return pd.NA

def dist_maker_df(df: pd.DataFrame):
    dists = {'metro_dist': (0.0, 0.0),
 'domodedovo_dist': (55.425398, 37.892624),
 'sherem_dist': (55.979653, 37.414202),
 'kremlin_dist': (55.752004, 37.617734),
             'moskva_city_dist': (55.749776, 37.536234),
             'mgu_dist': (55.703589, 37.530797),
             'nekrasovka_dist': (55.704123, 37.926320),
             'strogino_dist': (55.800774, 37.416551),
 'izmaylovo_dist': (55.792827, 37.76225)
            }
    
    for place, coord in tqdm(dists.items()):
        
        if place == 'metro_dist':
            
            df[place] = [dist(c1, c2) for c1, c2 in df[['coord_house','coords']].itertuples(index=False)]
        else:
            
            df[place] = [dist(c1, coord) for c1 in df['coord_house']]



###################################################################################################################################################################################


#CUSTOM IMPUTERS


class Coord_imp(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()
        X_copy['coord_house'] = tuple(zip(X_copy['latitude'].astype(float), X_copy['longitude'].astype(float)))

        return X_copy

class Metro_Names(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):

        def close(coords):
    
            metro_ = metro.copy()
            metro_['dist'] = [dist(coords, metro_i) for metro_i in metro_['coords']]
            return metro_.loc[metro_['dist']==min(metro_['dist'])].metro.to_list()[0]
        
        X_copy = X.copy()
        X_copy['metro_station'] = [close(coords) if name == 'Нет данных' else name for coords, name in X_copy[['coord_house','metro']].itertuples(index=False)]

        return X_copy

class Metro_Normalizer(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        
        X_copy = X.copy()
        metro_copy = metro.copy()
        

        X_copy['metro_m']=X_copy['metro_station'].map(strr)
        metro_copy['metro_m']=metro_copy['metro'].map(strr)

        metro_copy = metro_copy.drop_duplicates(subset=['metro'])

        X_copy_2 = pd.merge(X_copy, metro_copy, how='left', on='metro_m')

        return X_copy_2


class Dist_Maker(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):

        
        X_copy = X.copy()

        dist_maker_df(X_copy)

        return X_copy


class SmartDropper(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()
    
        drop_ = ['offer_type', 'city','street', 'house', 'phone','raion',
               'living_area', 'kitchen_area', 'all_rooms_area','house_material_type',
               'edit_time','house_material_bti', 'is_emergency', 'house_overlap_type', 'metro_x','metro_y',
               'metro_time', 'travel_type', 'coord_house', 'metro_station']
        
        X_copy = X_copy.drop(drop_, axis=1)

        if 'loggiascount' in X_copy.columns:
            X_copy = X_copy.drop(['loggiascount'], axis=1)
        if 'loggiasCount' in X_copy.columns:
            X_copy = X_copy.drop(['loggiasCount'], axis=1)
        if 'room_type' in X_copy.columns:
            X_copy = X_copy.drop(['room_type'], axis=1)
            


        return X_copy

class Some_transformations(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()
        def heat_func(x):
            if x == 'Нет данных':
                return 'central'
            return x
        
        X_copy['house_heating_supply'] = X_copy['house_heating_supply'].map(heat_func)

        def apartment(x):
            if x in ['Нет данных','False',False]:
                return 0
            return 1
            
        X_copy['is_apartment'] = X_copy['is_apartment'].map(apartment)

        cur = {'rub':1,
                   'rur':1,
                  'usd': usd,
                  'eur' : eur,
                  'euro' : eur}
        
        X_copy['currency'] = X_copy['currency'].map(cur)
        X_copy['price'] = X_copy['price'].astype(float) *  X_copy['currency'].astype(float)
            

        return X_copy


class NA_OUT(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        
        X_copy = X.copy()

        X_copy = X_copy.dropna(subset=['coords'])

        return X_copy



class LargeImputers(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()

        def rooms(area):
            if area <= 40:
                return 1
            elif area <= 80:
                return 2
            elif area <= 120:
                return 3
            elif area <= 160:
                return 4
            return 5
        
        X_copy['total_area'] = X_copy['total_area'].astype(float)
        
        X_copy['room_count'] = [rooms(area) if np.isnan(room) or room in [0,'0','False',False,'Нет данных'] else room for area, room in X_copy[['total_area','room_count']].itertuples(index=False)]


        ############################################################################################################################
        # ГОД
        ############################################################################################################################

        def year(year):
            if pd.isna(year) or year == 'Нет данных' or year == 0:
                return pd.NA
            else:
                return int(year)
                
        X_copy['build_year'] = X_copy['build_year'].map(year)

        def year_inp(x):
            try:
                return metro_year.loc[metro_year.index == x].iloc[0].item()
            except:
                return 0
        
        
        X_copy['build_year'] = [year_inp(station) if pd.isna(year) or year <= 1500 or year in ['Нет данных ',0,False] else year for year, station in X_copy[['build_year','metro_m']].itertuples(index=False)]


        ############################################################################################################################
        # ГОД
        ############################################################################################################################
        return X_copy


class Material_IMP(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()

        def typer(year):
            if year <= 1963:
                return 'brick'
            elif year <= 1995:
                return 'panel'
            return 'monolith'
        
        X_copy['material_type'] = [typer(year) if mtype in [0,'0','False',False,'Нет данных'] or pd.isna(mtype) else mtype for year, mtype in X_copy[['build_year','material_type']].itertuples(index=False)]

        return X_copy


class Okrug_Normalizer(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()

        def func(station):
            okr = metro_okrug.loc[metro_okrug.metro_m == station].new_okrug.to_list()[0]
            if type(okr)==str:
                return okr
            return okr[0]

        X_copy.okrug = [func(station) if "АО" not in distr else distr for station, distr in tqdm(X_copy[['metro_m','okrug']].itertuples(index=False))]
        return X_copy


class Ceiler(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):

        X_copy = X.copy()
        
        ce = []
        repa = []

        for year, okrug, maty, he, rep, price, area in X_copy[['build_year','okrug','material_type','ceiling_height','repair_type','price','total_area']].itertuples(index=False):
            he = float(he)
            if 1.7 <= he <= 10:
                ce.append(he)
            else:
                f1 = CE.okrug == okrug
                f2 = CE.material_type == maty
                try:
                    x = CE.loc[f1&f2].ceiling_height.to_list()[0]
                except:
                    x = 2.8
        
                ce.append(x)
                    
            if pd.isna(rep) or rep in ['Нет данных',0,'0',False,'X','ХЗ','Х']:
                if year >= 2024:
                    repa.append('no')
                else:
                    f1 = RT.okrug == okrug
                    f2 = RT.prm.map(lambda x: price/area in x)
                    try:
                        x = RT.loc[f1&f2].repair_type.to_list()[0]
                    except:
                        x = 'cosmetic'

                    if type(x) != str:
                        x = x[0]
                        
                    repa.append(x)
            else:
                repa.append(rep)
                
        X_copy.ceiling_height = ce
        X_copy.repair_type = repa
        
        return X_copy

###################################################################################################################################################################
#PIPELINES

#1

#########################################################################
coord_list = ['latitude','longitude']
metro_list = ['coord_house','metro']
final_stage = ['house_heating_supply','is_apartment','currency','price']

first_step = ColumnTransformer(
    transformers=[
        ('house_coord', Coord_imp(), coord_list),
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')

second_step = ColumnTransformer(
    transformers=[
        ('metro_names', Metro_Names(), metro_list),
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')

third_step = ColumnTransformer(
    transformers=[
        ('metro_normalize', Metro_Normalizer(), make_column_selector()),
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')


fourth_step = ColumnTransformer(
    transformers=[
        ('dist_maker', Dist_Maker(), make_column_selector()),
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')


fifth_step = ColumnTransformer(
    transformers=[
        ('dropper', SmartDropper(), make_column_selector()),
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')

sixth_step = ColumnTransformer(
    transformers=[
        ('some_ltl_transformations', Some_transformations(), final_stage),
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')


preprocessor = Pipeline(
    [
        ('coord_maker', first_step),
        ('metro_names_maker', second_step),
        ('metro_normalizer', third_step),
        ('distance_maker', fourth_step),
        ('smart_dropper', fifth_step),
        ('little_transforms', sixth_step),
    ]
)
#########################################################################

#2

last_stage = ['total_area','room_count','build_year','metro_m']

first_step = ColumnTransformer(
    transformers=[
        ('diffrent_imputers_1', LargeImputers(), last_stage),
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')

second_stage = ColumnTransformer(
    transformers=[
        ('drop', 'drop', ['publication_date']),
        ('diffrent_imputers_2', Material_IMP(), ['build_year','material_type']),
        ('okrug_norm', Okrug_Normalizer(), ['okrug','metro_m']),
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')

third_stage = ColumnTransformer(
    transformers=[
        ('Ceil', Ceiler(), ['okrug','material_type','ceiling_height','repair_type','build_year','price','total_area'])
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')


preprocessor_stage_2 = Pipeline(
    [
        ('some large transformations', first_step),
        ('drop and material', second_stage),
        ('ceiler', third_stage)
    ]
)

###################################################################################################################################################################

#PREPROCESSING


from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper



class Preprocessor():
    
    def __init__(self, path_or_DF):

        global metro, metro_year, metro_okrug, CE, RT

        metro = pd.read_pickle('data/metro.pkl')
        metro_year = pd.read_pickle('data/metro_year.pkl')
        metro_okrug = pd.read_pickle('data/metro_okrug.pkl')
        RT = pd.read_pickle('data/okrug_to_repair.pkl')
        CE = pd.read_pickle('data/okgug_ceil_repair.pkl')
        

        if type(path_or_DF) == str:
            self.path = path_or_DF
            self.data = pd.read_csv(self.path).drop_duplicates(subset=['id']).set_index('id')
        else:
            self.data = path_or_DF.drop_duplicates(subset=['id']).set_index('id')
        
    @timeit
    def transform(self):

        self.data_pre = preprocessor.fit_transform(self.data)
        self.data_pre.index = self.data.index.to_list()
        self.data_pre = self.data_pre.dropna(subset=['coords']).drop(['currency','coords'],axis=1)

        if 'room_count' not in tqdm(self.data_pre.columns):
            self.data_pre['room_count'] = np.NaN

        self.data_pre = preprocessor_stage_2.fit_transform(self.data_pre)
        self.data_pre = self.data_pre[self.data_pre.build_year != 0]
        self.data_pre.metro_dist = self.data_pre.metro_dist.astype(float)

    def to_pandas(self):

        return self.data_pre

    def save(self, title = None):

        if title is None:
            ct = datetime.datetime.now()
            title = f'data_{str(ct)[:str(ct).find(".")].replace(" ","_")}.pkl'

        try:
            self.data_pre.to_pickle(title)
            print(f'Записи обработаны. Файл сохраненен как {title}')
        except:
            
            print('You should call meth. transform first')





###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
######################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
######################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################













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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from category_encoders import CatBoostEncoder



class Nummer(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()
        return X_copy.astype(float)

class Booler(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()
        return X_copy.astype(bool)


class ReSHAP(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        
        X_copy = X.copy()
        
        X_copy = X_copy[['latitude','longitude','total_area','moskva_city_dist','kremlin_dist','repair_type','metro_m']]
        
        return X_copy






cat_f = ['material_type',
 'okrug',
 'metro_m',
 'repair_type',
 'house_heating_supply',
 'flat_type',
 'line']

num_f = ['latitude',
 'longitude',
 'combined_wcs_count',
 'floor',
 'floors_count',
 'metro_dist',
 'is_penthause',
 'build_year',
 'total_area',
 'room_count',
 'ceiling_height',
 'is_apartment',
 'domodedovo_dist',
 'sherem_dist',
 'kremlin_dist',
 'moskva_city_dist',
 'mgu_dist',
 'nekrasovka_dist',
 'strogino_dist',
 'izmaylovo_dist']



str_to_num = ['latitude','longitude','combined_wcs_count','floor','floors_count']
str_to_bool = ['is_penthause']

x_to_num = ColumnTransformer(
    transformers=[
        ('Nummer', Nummer(), str_to_num),
        ('Booler', Booler(), str_to_bool)
        
        
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')





scaler_encoder = ColumnTransformer(
    transformers=[
        ('Encode', CatBoostEncoder(), cat_f),
        ('Scale',StandardScaler(), num_f)
        
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')


preprocessor_2 = Pipeline(
    [
        ('num_to_num',x_to_num),
        ('Encoding_and_scaling',scaler_encoder),
    ]
)

from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error as rmse, mean_absolute_error as mae, r2_score




cat = CatBoostRegressor(random_state=42, verbose=False)
ml_pipeline = Pipeline(
    [
        ('preprocessor', preprocessor_2),
        ('catboost', cat)
    ])




scaler_encoder_l = ColumnTransformer(
    transformers=[
        ('Encode', CatBoostEncoder(), [5,6]),
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')


preprocessor_l = Pipeline(
    [
        ('Encoding_and_scaling',scaler_encoder_l),
    ])

ml_pipeline_l = Pipeline(
    [
        ('preprocessor', preprocessor_l),
        ('catboost', CatBoostRegressor(verbose=False, random_state=42))
    ])





from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


class To_Model_BIG():

    def __init__(self, path_or_DF, k=1.75, test_size = 0.2):

        if type(path_or_DF) == str:
            
            self.path = path_or_DF
            data = pd.read_pickle(self.path)

            
            self.X, self.Y = data.drop('price', axis = 1), data['price']
            
        else:
            
            self.X, self.Y = path_or_DF.drop('price', axis = 1), path_or_DF['price']


        try:
            self.X = self.X.drop(['room_type','loggiascount'], axis=1)
        except:
            pass
        
        self.k = k
        self.test_size = test_size

        
        self.pipe = ml_pipeline
        self.pipe_ltl = ml_pipeline_l


        ss = StandardScaler()
        ss.fit(pd.DataFrame(np.log(self.Y)))
        
        yy = ss.transform(pd.DataFrame(np.log(self.Y)))

        f =  ((-self.k <= yy.price) & (yy.price <= self.k))
        y_p = yy.loc[f]

        self.X_, self.Y_ = self.X.loc[y_p.index], self.Y[y_p.index]

        self.K_INX = len(y_p.index)/len(self.Y)

    @timeit
    def fit(self):

        X_train, X_test, y_train, self.y_test = train_test_split(self.X_, self.Y_, test_size=self.test_size, random_state=42)

        self.pipe.fit(X_train,y_train)

        self.y_pred = np.round(self.pipe.predict(self.X),-5)

        self.y_pred_V = np.round(self.pipe.predict(X_test),-5)

        self.X['pred_price'] = self.y_pred.astype(int)
        self.X['price'] = self.Y

        X_ = X_train[['latitude','longitude','total_area','moskva_city_dist','kremlin_dist','repair_type','metro_m']]
        X_[['latitude','longitude','total_area','moskva_city_dist','kremlin_dist']] = X_[['latitude','longitude','total_area','moskva_city_dist','kremlin_dist']].astype(float)
        
        self.pipe_ltl.fit(X_,y_train)

    def save(self, base = None, weights = None, weights_l=None):


        if weights is None:
            weights = 'model.pkl'

        if base is None:
            base = 'cian_base.pkl'

        if weights_l is None:
            weights_l = 'model_ltl.pkl'

        try: 
            pickle.dump(self.pipe, open(weights, 'wb'))
            print(f'Веса модели сохранены. Файл сохраненен как {weights}')
            self.X.to_pickle(base)
            print(f'Предсказанные цены добавлены в файл. Файл сохраненен как {base}')
            pickle.dump(self.pipe_ltl, open(weights_l, 'wb'))
            print(f'Веса модели mini сохранены. Файл сохраненен как {weights_l}')
            
        except:
            print('You should call meth. transform first')
            

    def score(self):

        rmse_ = rmse(y_pred=self.y_pred_V, y_true=self.y_test)

        mae_ = mae(y_pred=self.y_pred_V, y_true=self.y_test)
        
        r2 = r2_score(y_pred=self.y_pred_V, y_true=self.y_test)

        text = f'NO LOG: RMSE:{np.round(rmse_)}, MAE:{np.round(mae_)}, R2-score:{np.round(r2,4)}\nИндекс покрытия: {np.round(self.K_INX,2)}'

        print(text)


    def to_pandas(self):
        return self.X


    def load_predict_BIG(self, X: pd.DataFrame, weights=None):
        if weights is None:
            weights = 'model.pkl'
            
        self.model = pickle.load(open(weights, 'rb'))

        pred = self.model.predict(X)

        return pred


    def load_predict_small(self, X: pd.DataFrame, pipe=None):
        if pipe is None:
            pipe = 'model_ltl.pkl'
            
        self.model_l = pickle.load(open(weights, 'rb'))

        pred = self.model_l.predict(X)

        return pred

















































import cloudscraper
from math import ceil
import re
import json
import random
from bs4 import BeautifulSoup
import time
from fake_headers import Headers

def get_headers():

    """
    Generate headers for HTTP requests.

    Returns:
        dict[str, str]: Dictionary of headers.
    """

    headers = {
        'authority': 'cian.ru',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-encoding': 'gzip, deflate, br, zstd',
        'refer': 'https://www.google.com/',
        'accept-language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'cache-control': 'max-age=0',
        'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        'sec-ch-ua-arch': '"x86"',
        'sec-ch-ua-bitness': '"64"',
        'sec-ch-ua-full-version': '"115.0.5790.110"',
        'sec-ch-ua-full-version-list': '"Not/A)Brand";v="99.0.0.0", "Google Chrome";v="115.0.5790.110", "Chromium";v="115.0.5790.110"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-model': '""',
        'sec-ch-ua-platform': 'Windows',
        'sec-ch-ua-platform-version': '15.0.0',
        'sec-ch-ua-wow64': '?0',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': Headers(browser='chrome', os='win').generate()['User-Agent'],
        'x-client-data': '#..',
    }
    return headers

class Parser():

    def __init__(self, proxy=None, url=None):
        if url == None:
            self.url = 'https://www.cian.ru/cat.php?currency=2&deal_type=sale&engine_version=2&offer_type=flat&region=1&totime=-2'
        else:
            self.url = url
            
        self.proxy = proxy
        self.scraper = cloudscraper.create_scraper()

    def build_url(self, rooms, min_price, max_price, p):
        rooms = "".join([f"&room{room}=1" for room in rooms])
        price = f"&minprice={min_price}&maxprice={max_price}"
        url = self.url + f"&p={p}"+ rooms + price
        return url

    def get_offers(self, rooms, min_price, max_price, pages=54):
        urls = set()

        find_n_pages = True

        page_counter = 1

        while page_counter <= pages:
            
            print(f"Parsing page {page_counter} out of {pages} pages")
            print(f"{len(urls)} urls found already")
            url = self.build_url(rooms, min_price, max_price, page_counter)
            print(f"Scraping {url}")
            random_proxy = random.choice(self.proxy)
            headers = get_headers()
            
            n_attempts = 0

            while n_attempts < 3:
                try:
                    response = self.scraper.get(url, headers=headers, proxies=random_proxy)
                    html = response.text
                    if find_n_pages:
                        try:
                            pages = self.get_n_pages(html)
                            find_n_pages = False
                        except Exception as e:
                            print(f"Failed to get number of pages because of : {e}")

                    if response.status_code == 429:
                        time.sleep(15)

                    
                    elif response.status_code == 200:
                        bs = BeautifulSoup(html, 'html.parser')
                        url_tags = bs.find_all('div', {'data-name': 'GeneralInfoSectionRowComponent'})
                        for tag in url_tags:
                            url_tag = tag.find('a')
                            if url_tag:
                                try:
                                    link = url_tag['href']
                                    if 'www.cian.ru/sale' in link:
                                        urls.add(link)
                                        n_attempts = 3
                                except KeyError:
                                    pass
                    n_attempts += 1

                except Exception as e:
                    print('Error: ', e)
                    n_attempts += 1
                                
            page_counter += 1
        print(f"Collected {len(urls)} urls to parse")
        return urls           

    def get_n_pages(self, html):
        bs = BeautifulSoup(html, "html.parser")
        n_adds = bs.find_all('div', {'data-name': 'SummaryHeader'})[0].find('h5').text
        n_pages = min(54, ceil(int(re.sub(r'\D', '', n_adds))/28))
        return n_pages


    def extract_data(self, data, keys):
        try:
            value = data
            for key in keys:
                value = value[key]
            return value
        except (KeyError, IndexError):
            return None


    def get_data(self, urls, max_tries=3):
        scraped_data = []
        for url in urls:
            n_attempts = 0
            while n_attempts < max_tries:
                try:
                    # proxy = random.choice(self.proxy)
                    headers = get_headers()
                    response = self.scraper.get(url=url, headers=headers)#, proxies=proxy)
                    print(f"Trying to parse {url}, response status: {response.status_code}")
                    if response.status_code == 429:
                            time.sleep(15)
                    elif response.status_code == 200:
                            html = response.text
                            data = self.parse_data(html)
                            scraped_data.append(data)
                            break
                except Exception as e:
                    print("Failed attempt to get data because of : ", e)
                n_attempts += 1
        print(f"Amount of aquired data: {len(scraped_data)}")
        return scraped_data

    def parse_data(self, html):
        bs = BeautifulSoup(html, "html.parser")
        scripts = bs.find_all("script")

        for script in scripts:
            if "window._cianConfig['frontend-offer-card']" in script.text:
                start_index = script.text.find('concat([') + 7
                end_index = script.text.rfind(']') + 1
                json_data = script.text[start_index:end_index]
                parsed_json = json.loads(json_data)
                data_index = len(parsed_json) - 1
                while data_index >= 0:
                    try:
                        parsed_json[data_index]['value']['offerData']
                        break
                    except (TypeError, KeyError):
                        data_index -= 1

                data = {}

                data['offer_type'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'offerType']) or 'Нет данных'
                data['city'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'geo', 'address', 0, 'fullName']) or 'Нет данных'
                data['okrug'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'geo', 'address', 1, 'fullName']) or 'Нет данных'
                data['raion'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'geo', 'address', 2, 'fullName']) or 'Нет данных'
                data['street'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'geo', 'address', 3, 'fullName']) or 'Нет данных'
                data['house'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'geo', 'address', 4, 'fullName']) or 'Нет данных'
                data['room_count'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'roomsCount']) or 0
                data['room_type'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'roomType']) or 'Нет данных'
                data['loggiasCount'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'loggiasCount']) or 0
                data['latitude'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'geo', 'coordinates', 'lat']) or 0.0
                data['longitude'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'geo', 'coordinates', 'lng']) or 0.0
                data['id'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'id']) or 0
                data['phone'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'phones', 0, 'countryCode']) + self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'phones', 0, 'number']) or 0
                data['flat_type'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'flatType']) or 'Нет данных'
                data['is_apartment'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'isApartments']) or False
                data['is_penthause'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'isPenthouse']) or False
                data['total_area'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'totalArea']) or 0
                data['living_area'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'livingArea']) or 0
                data['kitchen_area'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'kitchenArea']) or 0
                data['all_rooms_area'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'allRoomsArea']) or 0
                data['combined_wcs_count'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'combinedWcsCount']) or 0
                data['repair_type'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'repairType']) or 'Нет данных'
                data['floor'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'floorNumber']) or 0
                data['price'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'bargainTerms', 'price']) or 0
                data['currency'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'bargainTerms', 'currency']) or 'Нет данных'
                data['material_type'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'building', 'materialType']) or 'Нет данных'
                data['floors_count'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'building', 'floorsCount']) or 0
                data['build_year'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'building', 'buildYear']) or 0
                data['ceiling_height'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'building', 'ceilingHeight']) or 0
                data['house_material_type'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'building', 'houseMaterialType']) or 'Нет данных'
                data['edit_time'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'editDate']) or 0
                data['publication_date'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'publicationDate']) or 0
                data['house_material_bti'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'bti', 'houseData', 'houseMaterialType']) or 'Нет данных'
                data['house_heating_supply'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'bti', 'houseData', 'houseHeatSupplyType']) or 'Нет данных'
                data['is_emergency'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'bti', 'houseData', 'isEmergency']) or False
                data['house_overlap_type'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'bti', 'houseData', 'houseOverlapType']) or 'Нет данных'
                data['metro'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'geo', 'undergrounds', 0, 'name']) or 'Нет данных'
                data['metro_time'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'geo', 'undergrounds', 0, 'travelTime']) or 0
                data['travel_type'] = self.extract_data(parsed_json, [data_index, 'value', 'offerData', 'offer', 'geo', 'undergrounds', 0, 'travelType']) or 'Нет данных'

        return data

    def parse(self, rooms, min_price, max_price):
        urls = self.get_offers(rooms, min_price, max_price)
        data = self.get_data(urls)
        print(f"Collected {len(data)} real estate adds data")
        return data