import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


from funcions.cian import findr
from funcions.model_func import Result_Maker



app = FastAPI()



# Load model at startup
@app.on_event("startup")
def startup_event():
    # global model
    # model = load_model()
    pass

@app.get('/')
def return_info():
    return 'Hello, stranger. Welcome to CIAN-based ML-service!'

#### /cian_id
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


#### /classify
class Item(BaseModel):
    square: int
    quality: str
    lat: float
    lan: float
class ClassifyClass(BaseModel):
    classify_me: str
    metro_m : str


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
        metro_m     = str(result[-1]))
    
    return response


#### /classify_max
class ItemMax(BaseModel):
    total_area: int
    repair_type: str
    lat: float
    lan: float
    material_type: str
    floor: int
    floors_count: int
    building_year: int

class ClassifyClass(BaseModel):
    classify_me: str
    metro_m : str


@app.post('/classify_max')
def classify(data: ItemMax):

    total_area = int(data.total_area)
    repair_type = data.repair_type
    lat = float(data.lat)
    lan = float(data.lan)



    # json = {'total_area':total_area,'repair_type':repair_type,'lat':lat,'lan':lan}

    # result = Result_Maker(json)



    # response = ClassifyClass(
    #     classify_me = str(result[0]),
    #     metro_m     = str(result[-1]))
    
    return f'Пока что тут ничего нет. Но когда-нибудь...'

##### run from api folder:
##### uvicorn app.main:app
##### export PYTHONPATH=$PYTHONPATH:/