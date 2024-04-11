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
 'room_type',
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
 'loggiascount',
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


preprocessor = Pipeline(
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
        ('preprocessor', preprocessor),
        ('catboost', cat)
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

        
        self.k = k
        self.test_size = test_size

        
        


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

        ml_pipeline.fit(X_train,y_train)

        self.y_pred = np.round(ml_pipeline.predict(self.X),-5)

        self.y_pred_V = np.round(ml_pipeline.predict(X_test),-5)

        self.X['pred_price'] = self.y_pred.astype(int)
        self.X['price'] = self.Y

    def save(self, title = None):

        if title is None:
            title = 'cian_base.pkl'

        try:
            try: 
                self.X.to_pickle(title)
                print(f'Предсказанные цены добавлены в файл. Файл сохраненен как {title}')
            except:
                self.X.to_pickle(title+'.pkl')
                print(f'Предсказанные цены добавлены в файл. Файл сохраненен как {title}')
            
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

        
