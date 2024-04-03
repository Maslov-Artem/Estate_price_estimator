from fastapi import FastAPI, File, UploadFile, Depends
from pydantic import BaseModel
from utils.model_funcs import f

def load_model(*args):


    if len(args) == 4:

        return args[0]*350000
    return None

model = None 
app = FastAPI()



# Create class of answer: only class name 
class FindClass(BaseModel):
    find_me: str

class CianClass(BaseModel):
    id : int
    link : str
    price : float
    base : float

   
class Item(BaseModel):
    square: str
    quality: str
    lat: float
    lan: float 

class ClassifyClass(BaseModel):
    classify_me: float

# Load model at startup
@app.on_event("startup")
def startup_event():
    global model
    model = load_model

@app.get('/')
def return_info():
    return 'Hello FastAPI'


@app.post('/classify')
def classify(data: Item):

    square = int(data.square)
    quality = data.quality
    lat = float(data.lat)
    lan = float(data.lan)

    result = float(model(square, quality, lat, lan))
    response = ClassifyClass(
        classify_me = result)

    return response

@app.post('/cian_id')
def clf_text(data: FindClass):
    id_ = int(data.find_me)

    ###
    ###
    ###


    link_ = f'cian.ru/sale/flat{id}'
    price_ = 10
    base_ = 12
 
    response = CianClass(
        id = id_,
        link = link_,
        price= price_,
        base= base_
        )

    return response


##### run from api folder:
##### uvicorn app.main:app