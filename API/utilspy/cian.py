from functools import cache
import pandas as pd

def load_base():
    df = pd.read_pickle('data/cian_base.pkl')
    return df

@cache
def findr(id):

    

    df = load_base()

    json = {'total_area':-1,'repair_type':'No data','lat':-1,'lan':-1,'metro_dist':-1,'metro':'-1','kremlin_dist':-1,'pred':-1,'real':-1, 'link': 'No data'}

    try:
        id = int(id)
        f = df.index == id

        total_area = df.loc[f].iloc[:,4].item()
        repair_type = df.loc[f].iloc[:,7].item()
        lat = df.loc[f].iloc[:,10].item()
        lan = df.loc[f].iloc[:,11].item()
        
        metro_dist = df.loc[f].iloc[:,18].item()
        kremlin_dist = df.loc[f].iloc[:,21].item()
        pred = df.loc[f].iloc[:,27].item()
        real = df.loc[f].iloc[:,28].item()
        metro = df.loc[f].iloc[:,3].item()

        link = f'cian.ru/sale/flat/{id}'
        json = {'total_area':float(total_area),'repair_type':str(repair_type),'lat':float(lat),'lan':float(lan),'metro_dist':float(metro_dist),'metro':str(metro),'kremlin_dist':float(kremlin_dist),'pred':float(pred),'real':float(real), 'link': str(link)}
    
        return json
    except:
        return json