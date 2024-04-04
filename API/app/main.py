from fastapi import FastAPI
from pydantic import BaseModel
from utilspy.model_func import Result_Maker, Nummer 
from utilspy.cian import findr
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

app = FastAPI()



# Create class of answer: only class name 
class FindClass(BaseModel):
    find_me: str

class CianClass(BaseModel):
        total_area : float
        repair_type : str
        lat : float
        lan : float
        metro_dist : float
        metro : str
        kremlin_dist : float
        pred : float
        real : float
        link : str

   
class Item(BaseModel):
    square: int
    quality: str
    lat: float
    lan: float


class ClassifyClass(BaseModel):
    classify_me: str
    metro_m : str

# Load model at startup
@app.on_event("startup")
def startup_event():
    # global model
    # model = load_model()
    pass

@app.get('/')
def return_info():
    return 'Hello FastAPI'

@app.post('/cian_id')
def clf_text(data: FindClass):

    json = findr(data.find_me)
    
    response = CianClass(
        total_area = json['total_area'],
        repair_type = json['repair_type'],
        lat = json['lat'],
        lan = json['lan'],
        metro_dist= json['metro_dist'],
        metro= json['metro'],
        kremlin_dist = json['kremlin_dist'],
        pred = json['pred'],
        real = json['real'],
        link = json['link'],
        )

    return response


@app.post('/classify')
def classify(data: Item):

    square = int(data.square)
    quality = data.quality
    lat = float(data.lat)
    lan = float(data.lan)

    json = {'square':square,'quality':quality,'lat':lat,'lan':lan}

    result = Result_Maker(json)


    response = ClassifyClass(
        classify_me = str(result[0]),
        metro_m     = str(result[1])
        )

    return response

##### run from api folder:
##### uvicorn app.main:app