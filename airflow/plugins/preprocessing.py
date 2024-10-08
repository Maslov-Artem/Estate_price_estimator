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
 'izmaylovo_dist': (55.792827, 37.76225)}
    
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
        
        drop_1 = ['offer_type', 'city','street', 'house','phone','raion',
               'room_type', 'loggiasCount',
               'living_area', 'kitchen_area', 'all_rooms_area', 'house_material_type',
               'edit_time','house_material_bti', 'is_emergency', 'house_overlap_type', 'metro_x','metro_y',
               'metro_time', 'travel_type', 'coord_house', 'metro_station']

        drop_2 = ['offer_type', 'city','street', 'house', 'phone','raion',
               'living_area', 'kitchen_area', 'all_rooms_area','house_material_type',
               'edit_time','house_material_bti', 'is_emergency', 'house_overlap_type', 'metro_x','metro_y',
               'metro_time', 'travel_type', 'coord_house', 'metro_station']

        

        try:
            X_copy = X_copy.drop(drop_1, axis=1)
        except:
            X_copy = X_copy.drop(drop_2, axis=1)


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
        # ПОТОЛОК
        ############################################################################################################################
        height = X_copy.loc[X_copy.ceiling_height != 'Нет данных'].ceiling_height.astype(float).median()

        def ceil(x):
            if x == 'Нет данных':
                return height
            x = float(x)

            if x <= 1.5:
                return height
            return x

        X_copy['ceiling_height'] = X_copy['ceiling_height'].map(ceil)

        ############################################################################################################################
        # ПОТОЛОК
        ############################################################################################################################

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

        def repair_imp(x):
            if pd.isna(x) or x in ['Нет данных',0,'0',False]:
                return 'no'
            return x
        
        X_copy['repair_type'] = X_copy['repair_type'].map(repair_imp)
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

last_stage = ['total_area','room_count','ceiling_height','build_year','metro_m','repair_type']

first_step = ColumnTransformer(
    transformers=[
        ('diffrent_imputers', LargeImputers(), last_stage),
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')

second_stage = ColumnTransformer(
    transformers=[
        ('drop', 'drop', ['publication_date']),
        ('diffrent_imputers', Material_IMP(), ['build_year','material_type']),
        ('okrug_norm', Okrug_Normalizer(), ['okrug','metro_m']),
    ],
     verbose_feature_names_out = False,
     remainder = 'passthrough')


preprocessor_stage_2 = Pipeline(
    [
        ('some large transformations', first_step),
        ('drop and material', second_stage)
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
    
    def __init__(self, path):

        global metro, metro_year, metro_okrug

        metro = pd.read_pickle('/opt/airflow/plugins/metro.pkl')
        metro_year = pd.read_pickle('/opt/airflow/plugins/metro_year.pkl')
        metro_okrug = pd.read_pickle('/opt/airflow/plugins/metro_okrug.pkl')

        self.path = path
        self.data = pd.read_csv(path).drop_duplicates(subset=['id']).set_index('id')

        
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

            try: 
                self.data_pre.to_pickle(title)
                print(f'Записи обработаны. Файл сохраненен как {title}')
            except:
                self.data_pre.to_pickle(title+'.pkl')
                print(f'Записи обработаны. Файл сохраненен как {title}')
        except:
            
            print('You should call meth. transform first')
