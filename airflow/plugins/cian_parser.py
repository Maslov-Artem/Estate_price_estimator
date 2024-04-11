import cloudscraper

from math import ceil
import re
from helpers import get_headers
import json
import random
from bs4 import BeautifulSoup
import time

class Parser():

    small_apt_price_ranges = price_dict = {
    1000000: 5000000,
    5000001: 6000000,
    6000001: 6500000,
    6500001: 7000000,
    7000001: 7300000,
    7300001: 7500000,
    7500001: 7700000,
    7700001: 7900000,
    7900001: 8100000,
    8100001: 8300000,
    8300001: 8500000,
    8500001: 8700000,
    8700001: 8900000,
    8900001: 9050000,
    9050001: 9250000,
    9250001: 9450000,
    9450001: 9550000,
    9550001: 9750000,
    9750001: 9900000,
    9900001: 10000000,
    10000001: 10160000,
    10160001: 10300000,
    10300001: 10450000,
    10450001: 10550000,
    10550001: 10700000,
    10700001: 10800000,
    10800001: 10900000,
    10900001: 10990000,
    10990001: 11100000,
    11100001: 11250000,
    11250001: 11400000,
    11400001: 11500000,
    11500001: 11600000,
    11600001: 11750000,
    11750001: 11900000,
    11900001: 11990000,
    11990001: 12090000,
    12090001: 12250000,
    12250001: 12450000,
    12450001: 12550000,
    12550001: 12750000,
    12750001: 12950000,
    12950001: 13050000,
    13050001: 13250000,
    13250001: 13450000,
    13450001: 13600000,
    13600001: 13800000,
    13800001: 13900000,
    13900001: 14000000,
    14000001: 14100000,
    14100001: 14300000,
    14300001: 14400000,
    14400001: 14500000,
    14500001: 14600000,
    14600001: 14800000,
    14800001: 14900000,
    14900001: 15000000,
    15000001: 15200000,
    15200001: 15400000,
    15400001: 15500000,
    15500001: 15700000,
    15700001: 15950000,
    15950001: 16200000,
    16200001: 16450000,
    16450001: 16750000,
    16750001: 17050000,
    17050001: 17450000,
    17450001: 17750000,
    17750001: 18050000,
    18050001: 18450000,
    18450001: 18850000,
    18850001: 19300000,
    19300001: 19700000,
    19700001: 20000000,
    20000001: 20450000,
    20450001: 20950000,
    20950001: 21450000,
    21450001: 21950000,
    21950001: 22450000,
    22450001: 22950000,
    22950001: 23600000,
    23600001: 24500000,
    24500001: 25200000,
    25200001: 26200000,
    26200001: 27400000,
    27400001: 28800000,
    28800001: 30000000,
    30000001: 31500000,
    31500001: 33200000,
    33200001: 35500000,
    35500001: 39000000,
    39000001: 44000000,
    44000001: 54000000,
    54000001: 79000000,
    79000001: 999999999
    }

    big_apt_price_ranges = {
    1000001: 11000000,
    11000001: 12500000,
    12500001: 13400000,
    13400001: 14100000,
    14100001: 14700000,
    14700001: 15300000,
    15300001: 15800000,
    15800001: 16400000,
    16400001: 17000000,
    17000001: 17700000,
    17700001: 18400000,
    18400001: 19200000,
    19200001: 20000000,
    20000001: 21000000,
    21000001: 22000000,
    22000001: 23000000,
    23000001: 24000000,
    24000001: 25000000,
    25000001: 26000000,
    26000001: 27000000,
    27000001: 28000000,
    28000001: 29000000,
    29000001: 30000000,
    30000001: 32000000,
    32000001: 34000000,
    34000001: 36000000,
    36000001: 38000000,
    38000001: 40000000,
    40000001: 42000000,
    42000001: 44000000,
    44000001: 46000000,
    46000001: 49000000,
    49000001: 52000000,
    52000001: 57000000,
    57000001: 64000000,
    64000001: 70000000,
    70000001: 79000000,
    79000001: 90000000,
    90000001: 105000000,
    105000001: 125000000,
    125000001: 145000000,
    145000001: 185000000,
    185000001: 240000000,
    240000001: 340000000,
    340000001: 999999999,
    }


    small_apt_rooms =[1, 2, 9]

    big_apt_rooms = [3, 4, 5, 6]

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
                    proxy = random.choice(self.proxy)
                    headers = get_headers()
                    response = self.scraper.get(url=url, headers=headers, proxies=proxy)
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
