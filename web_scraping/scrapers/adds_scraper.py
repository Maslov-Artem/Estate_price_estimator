import json
import csv
import asyncio
from aiocfscrape import CloudflareScraper
from bs4 import BeautifulSoup
from fake_headers import Headers
import numpy as np
import random

import requests


def get_headers() -> dict:
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


async def fetch_data(session: CloudflareScraper, url: str, writer: csv.DictWriter, semaphore: asyncio.Semaphore, retry_attempts: int = 3) -> dict:
    """
    Fetch data from a URL asynchronously and write it to a CSV file.

    Args:
        session (CloudflareScraper): Async HTTP session.
        url (str): URL to fetch data from.
        writer (csv.DictWriter): CSV writer to write the data.
        semaphore (asyncio.Semaphore): Semaphore for controlling concurrent access.
        retry_attempts (int, optional): Number of retry attempts. Defaults to 3.

    Returns:
        dict: Dictionary containing fetched data.
    """
    for attempt in range(retry_attempts):
        try:
            proxies = ['http://661a66002b:bb68065ca2@95.31.211.120:30848',
                       'http://MTdPVT:1vaBRr@46.161.44.120:9252',
                       'http://MTdPVT:1vaBRr@46.161.47.209:9262',
                       'http://gJkm1m:Z4NDKQ@46.161.46.33:9003',
                       ]
            proxy = np.random.choice(proxies, p=[0.4, 0.2, 0.2, 0.2])
            if proxy == ['http://661a66002b:bb68065ca2@95.31.211.120:30848']:
                proxy_headers = {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'ru',
                    'Connection': 'keep-alive',
                    'Host': 'proxys.io',
                    'Referer': 'https://proxys.io/ru/mobileProxies/account/show?order=23051',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'same-origin',
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15'
                }
                requests.get(
                    'https://proxys.io/ru/api/v2/change-mobile-proxy-ip?key=65eb7ee74ffdaf25902d9cc487edcaae&order=23051&proxy=1', headers=proxy_headers)
            headers = get_headers()
            async with semaphore:
                async with session.get(url, proxy=proxy, headers=headers) as response:
                    if response.status == 429:
                        print(f'Proxy {proxy} is overheating')
                    if response.status == 200:
                        html = await response.text()
                        data = parse_data(html)
                        write_to_csv(writer, data)  # Write data to CSV
                        return data
                    else:
                        return None
        except Exception as e:
            print('Error: ', e)
            if attempt < retry_attempts - 1:
                # Exponential backoff before retrying
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print("Maximum retry attempts reached.")
                return None


def parse_data(data: str) -> dict:
    """
    Parse HTML data and extract relevant information.

    Args:
        data (str): HTML data to parse.

    Returns:
        dict: Dictionary containing parsed data.
    """
    bs = BeautifulSoup(data, 'html.parser')
    scripts = bs.find_all('script')

    for script in scripts:
        if "window._cianConfig['frontend-offer-card']" in script.text:
            start_index = script.text.find('concat([') + 7
            end_index = script.text.rfind(']') + 1
            json_data = script.text[start_index:end_index]
            parsed_json = json.loads(json_data)
            with open('TypeError.json', 'w') as f:
                json.dump(parsed_json, f, indent=2)

            # Initialize data dictionary
            data = {}

            # Extracting data with error handling
            try:
                data['offer_type'] = parsed_json[124]['value']['offerData']['offer']['offerType']
            except KeyError:
                data['offer_type'] = 'Нет данных'
            except TypeError:
                try:
                    data['offer_type'] = parsed_json[125]['value']['offerData']['offer']['offerType']
                except KeyError:
                    data['offer_type'] = 'Нет данных'

            try:
                data['city'] = parsed_json[124]['value']['offerData']['offer']['geo']['address'][0]['fullName']
            except (KeyError, IndexError):
                data['city'] = 'Нет данных'
            except TypeError:
                try:
                    data['city'] = parsed_json[125]['value']['offerData']['offer']['geo']['address'][0]['fullName']
                except (KeyError, IndexError):
                    data['city'] = 'Нет данных'

            try:
                data['okrug'] = parsed_json[124]['value']['offerData']['offer']['geo']['address'][1]['fullName']
            except (KeyError, IndexError):
                data['okrug'] = 'Нет данных'
            except TypeError:
                try:
                    data['okrug'] = parsed_json[125]['value']['offerData']['offer']['geo']['address'][1]['fullName']
                except (KeyError, IndexError):
                    data['okrug'] = 'Нет данных'

            try:
                data['raion'] = parsed_json[124]['value']['offerData']['offer']['geo']['address'][2]['fullName']
            except (KeyError, IndexError):
                data['raion'] = 'Нет данных'
            except TypeError:
                try:
                    data['raion'] = parsed_json[125]['value']['offerData']['offer']['geo']['address'][2]['fullName']
                except (KeyError, IndexError):
                    data['raion'] = 'Нет данных'

            try:
                data['street'] = parsed_json[124]['value']['offerData']['offer']['geo']['address'][3]['fullName']
            except (KeyError, IndexError):
                data['street'] = 'Нет'
            except TypeError:
                try:
                    data['street'] = parsed_json[125]['value']['offerData']['offer']['geo']['address'][3]['fullName']
                except (KeyError, IndexError):
                    data['street'] = 'Нет'

            try:
                data['house'] = parsed_json[124]['value']['offerData']['offer']['geo']['address'][4]['fullName']
            except (KeyError, IndexError):
                data['house'] = 'Нет'
            except TypeError:
                try:
                    data['house'] = parsed_json[125]['value']['offerData']['offer']['geo']['address'][4]['fullName']
                except (KeyError, IndexError):
                    data['house'] = 'Нет'

            try:
                data['room_count'] = parsed_json[124]['value']['offerData']['offer']['roomsCount']
            except (KeyError, IndexError):
                data['room_count'] = 0
            except TypeError:
                try:
                    data['room_count'] = parsed_json[125]['value']['offerData']['offer']['roomsCount']
                except (KeyError, IndexError):
                    data['room_count'] = 0

            try:
                data['room_type'] = parsed_json[124]['value']['offerData']['offer']['roomType']
            except (KeyError, IndexError):
                data['room_type'] = 'Нет'
            except TypeError:
                try:
                    data['room_type'] = parsed_json[125]['value']['offerData']['offer']['roomType']
                except (KeyError, IndexError):
                    data['room_type'] = 'Нет'

            try:
                data['loggiasCount'] = parsed_json[124]['value']['offerData']['offer']['loggiasCount']
            except (KeyError, IndexError):
                data['loggiasCount'] = 'Нет'
            except TypeError:
                try:
                    data['loggiasCount'] = parsed_json[125]['value']['offerData']['offer']['loggiasCount']
                except (KeyError, IndexError):
                    data['loggiasCount'] = 'Нет'

            try:
                data['latitude'] = parsed_json[124]['value']['offerData']['offer']['geo']['coordinates']['lat']
            except KeyError:
                data['latitude'] = 0.0
            except TypeError:
                try:
                    data['latitude'] = parsed_json[125]['value']['offerData']['offer']['geo']['coordinates']['lat']
                except KeyError:
                    data['latitude'] = 0.0

            try:
                data['longitude'] = parsed_json[124]['value']['offerData']['offer']['geo']['coordinates']['lng']
            except KeyError:
                data['longitude'] = 0.0
            except TypeError:
                try:
                    data['longitude'] = parsed_json[125]['value']['offerData']['offer']['geo']['coordinates']['lng']
                except KeyError:
                    data['longitude'] = 0.0

            try:
                data['id'] = parsed_json[124]['value']['offerData']['offer']['id']
            except KeyError:
                data['id'] = 'Нет данных'
            except TypeError:
                try:
                    data['id'] = parsed_json[125]['value']['offerData']['offer']['id']
                except KeyError:
                    data['id'] = 'Нет данных'

            try:
                data['phone'] = parsed_json[124]['value']['offerData']['offer']['phones'][0]['countryCode'] + \
                    parsed_json[124]['value']['offerData']['offer']['phones'][0]['number']
            except (KeyError, IndexError):
                data['phone'] = 0
            except TypeError:
                try:
                    data['phone'] = parsed_json[125]['value']['offerData']['offer']['phones'][0]['countryCode'] + \
                        parsed_json[125]['value']['offerData']['offer']['phones'][0]['number']
                except (KeyError, IndexError):
                    data['phone'] = 0

            try:
                data['flat_type'] = parsed_json[124]['value']['offerData']['offer']['flatType']
            except KeyError:
                data['flat_type'] = 'Нет данных'
            except TypeError:
                try:
                    data['flat_type'] = parsed_json[125]['value']['offerData']['offer']['flatType']
                except KeyError:
                    data['flat_type'] = 'Нет данных'

            try:
                data['is_apartment'] = parsed_json[124]['value']['offerData']['offer']['isApartments']
            except KeyError:
                data['is_apartment'] = False
            except TypeError:
                try:
                    data['is_apartment'] = parsed_json[125]['value']['offerData']['offer']['isApartments']
                except KeyError:
                    data['is_apartment'] = False

            try:
                data['is_penthause'] = parsed_json[124]['value']['offerData']['offer']['isPenthouse']
            except KeyError:
                data['is_penthause'] = False
            except TypeError:
                try:
                    data['is_penthause'] = parsed_json[125]['value']['offerData']['offer']['isPenthouse']
                except KeyError:
                    data['is_penthause'] = False

            try:
                data['total_area'] = parsed_json[124]['value']['offerData']['offer']['totalArea']
            except KeyError:
                data['total_area'] = 0
            except TypeError:
                try:
                    data['total_area'] = parsed_json[125]['value']['offerData']['offer']['totalArea']
                except KeyError:
                    data['total_area'] = 0

            try:
                data['living_area'] = parsed_json[124]['value']['offerData']['offer']['livingArea']
            except KeyError:
                data['living_area'] = 0
            except TypeError:
                try:
                    data['living_area'] = parsed_json[125]['value']['offerData']['offer']['livingArea']
                except KeyError:
                    data['living_area'] = 0

            try:
                data['kitchen_area'] = parsed_json[124]['value']['offerData']['offer']['kitchenArea']
            except KeyError:
                data['kitchen_area'] = 0
            except TypeError:
                try:
                    data['kitchen_area'] = parsed_json[125]['value']['offerData']['offer']['kitchenArea']
                except KeyError:
                    data['kitchen_area'] = 0

            try:
                data['all_rooms_area'] = parsed_json[124]['value']['offerData']['offer']['allRoomsArea']
            except KeyError:
                data['all_rooms_area'] = 0
            except TypeError:
                try:
                    data['all_rooms_area'] = parsed_json[125]['value']['offerData']['offer']['allRoomsArea']
                except KeyError:
                    data['all_rooms_area'] = 0

            try:
                data['combined_wcs_count'] = parsed_json[124]['value']['offerData']['offer']['combinedWcsCount']
            except KeyError:
                data['combined_wcs_count'] = 0
            except TypeError:
                try:
                    data['combined_wcs_count'] = parsed_json[125]['value']['offerData']['offer']['combinedWcsCount']
                except KeyError:
                    data['combined_wcs_count'] = 0

            try:
                data['repair_type'] = parsed_json[124]['value']['offerData']['offer']['repairType']
            except KeyError:
                data['repair_type'] = 'Нет данных'
            except TypeError:
                try:
                    data['repair_type'] = parsed_json[125]['value']['offerData']['offer']['repairType']
                except KeyError:
                    data['repair_type'] = 'Нет данных'

            try:
                data['floor'] = parsed_json[124]['value']['offerData']['offer']['floorNumber']
            except KeyError:
                data['floor'] = 'Нет данных'
            except TypeError:
                try:
                    data['floor'] = parsed_json[125]['value']['offerData']['offer']['floorNumber']
                except KeyError:
                    data['floor'] = 'Нет данных'

            try:
                data['price'] = parsed_json[124]['value']['offerData']['offer']['bargainTerms']['price']
            except KeyError:
                data['price'] = None
            except TypeError:
                try:
                    data['price'] = parsed_json[125]['value']['offerData']['offer']['bargainTerms']['price']
                except KeyError:
                    data['price'] = None

            try:
                data['currency'] = parsed_json[124]['value']['offerData']['offer']['bargainTerms']['currency']
            except KeyError:
                data['currency'] = 'Нет данных'
            except TypeError:
                try:
                    data['currency'] = parsed_json[125]['value']['offerData']['offer']['bargainTerms']['currency']
                except KeyError:
                    data['currency'] = 'Нет данных'

            try:
                data['material_type'] = parsed_json[124]['value']['offerData']['offer']['building']['materialType']
            except KeyError:
                data['material_type'] = 'Нет данных'
            except TypeError:
                try:
                    data['material_type'] = parsed_json[125]['value']['offerData']['offer']['building']['materialType']
                except KeyError:
                    data['material_type'] = 'Нет данных'

            try:
                data['floors_count'] = parsed_json[124]['value']['offerData']['offer']['building']['floorsCount']
            except KeyError:
                data['floors_count'] = 'Нет данных'
            except TypeError:
                try:
                    data['floors_count'] = parsed_json[125]['value']['offerData']['offer']['building']['floorsCount']
                except KeyError:
                    data['floors_count'] = 'Нет данных'

            try:
                data['build_year'] = parsed_json[124]['value']['offerData']['offer']['building']['buildYear']
            except KeyError:
                data['build_year'] = 'Нет данных'
            except TypeError:
                try:
                    data['build_year'] = parsed_json[125]['value']['offerData']['offer']['building']['buildYear']
                except KeyError:
                    data['build_year'] = 'Нет данных'

            try:
                data['ceiling_height'] = parsed_json[124]['value']['offerData']['offer']['building']['ceilingHeight']
            except KeyError:
                data['ceiling_height'] = 'Нет данных'
            except TypeError:
                try:
                    data['ceiling_height'] = parsed_json[125]['value']['offerData']['offer']['building']['ceilingHeight']
                except KeyError:
                    data['ceiling_height'] = 'Нет данных'

            try:
                data['house_material_type'] = parsed_json[124]['value']['offerData']['offer']['building']['houseMaterialType']
            except KeyError:
                data['house_material_type'] = 'Нет данных'
            except TypeError:
                try:
                    data['house_material_type'] = parsed_json[125]['value']['offerData']['offer']['building']['houseMaterialType']
                except KeyError:
                    data['house_material_type'] = 'Нет данных'

            try:
                data['edit_time'] = parsed_json[124]['value']['offerData']['offer']['editDate']
            except KeyError:
                data['edit_time'] = 'Нет данных'
            except TypeError:
                try:
                    data['edit_time'] = parsed_json[125]['value']['offerData']['offer']['editDate']
                except KeyError:
                    data['edit_time'] = 'Нет данных'

            try:
                data['publication_date'] = parsed_json[124]['value']['offerData']['offer']['publicationDate']
            except KeyError:
                data['publication_date'] = 'Нет данных'
            except TypeError:
                try:
                    data['publication_date'] = parsed_json[125]['value']['offerData']['offer']['publicationDate']
                except KeyError:
                    data['publication_date'] = 'Нет данных'

            try:
                data['house_material_bti'] = parsed_json[124]['value']['offerData']['bti']['houseData']['houseMaterialType']
            except KeyError:
                data['house_material_bti'] = 'Нет данных'
            except TypeError:
                try:
                    data['house_material_bti'] = parsed_json[125]['value']['offerData']['bti']['houseData']['houseMaterialType']
                except KeyError:
                    data['house_material_bti'] = 'Нет данных'

            try:
                data['house_heating_supply'] = parsed_json[124]['value']['offerData']['bti']['houseData']['houseHeatSupplyType']
            except KeyError:
                data['house_heating_supply'] = 'Нет данных'
            except TypeError:
                try:
                    data['house_heating_supply'] = parsed_json[125]['value']['offerData']['bti']['houseData']['houseHeatSupplyType']
                except KeyError:
                    data['house_heating_supply'] = 'Нет данных'

            try:
                data['is_emergency'] = parsed_json[124]['value']['offerData']['bti']['houseData']['isEmergency']
            except KeyError:
                data['is_emergency'] = 'Нет данных'
            except TypeError:
                try:
                    data['is_emergency'] = parsed_json[125]['value']['offerData']['bti']['houseData']['isEmergency']
                except KeyError:
                    data['is_emergency'] = 'Нет данных'

            try:
                data['house_overlap_type'] = parsed_json[124]['value']['offerData']['bti']['houseData']['houseOverlapType']
            except KeyError:
                data['house_overlap_type'] = 'Нет данных'
            except TypeError:
                try:
                    data['house_overlap_type'] = parsed_json[125]['value']['offerData']['bti']['houseData']['houseOverlapType']
                except KeyError:
                    data['house_overlap_type'] = 'Нет данных'

            try:
                data['metro'] = parsed_json[124]['value']['offerData']['offer']['geo']['undergrounds'][0]['name']
            except (KeyError, IndexError):
                data['metro'] = 'Нет данных'
            except TypeError:
                try:
                    data['metro'] = parsed_json[125]['value']['offerData']['offer']['geo']['undergrounds'][0]['name']
                except (KeyError, IndexError):
                    data['metro'] = 'Нет данных'

            try:
                data['metro_time'] = parsed_json[124]['value']['offerData']['offer']['geo']['undergrounds'][0]['travelTime']
            except (KeyError, IndexError):
                data['metro_time'] = 'Нет данных'
            except TypeError:
                try:
                    data['metro_time'] = parsed_json[125]['value']['offerData']['offer']['geo']['undergrounds'][0]['travelTime']
                except (KeyError, IndexError):
                    data['metro_time'] = 'Нет данных'

            try:
                data['travel_type'] = parsed_json[124]['value']['offerData']['offer']['geo']['undergrounds'][0]['travelType']
            except (KeyError, IndexError):
                data['travel_type'] = 'Нет данных'
            except TypeError:
                try:
                    data['travel_type'] = parsed_json[125]['value']['offerData']['offer']['geo']['undergrounds'][0]['travelType']
                except (KeyError, IndexError):
                    data['travel_type'] = 'Нет данных'

            return data

    # Return None if no suitable script found
    return None


def write_to_csv(writer: csv.DictWriter, data: dict) -> None:
    """
    Write data to a CSV file.

    Args:
        writer (csv.DictWriter): CSV writer to write the data.
        data (dict): Dictionary containing data to be written.
    """
    writer.writerow({
        'offer_type': data['offer_type'],
        'city': data['city'],
        'okrug': data['okrug'],
        'raion': data['raion'],
        'street': data['street'],
        'house': data['house'],
        'room_count': data['room_count'],
        'room_type': data['room_type'],
        'loggiasCount': data['loggiasCount'],
        'latitude': data['latitude'],
        'longitude': data['longitude'],
        'id': data['id'],
        'phone': data['phone'],
        'flat_type': data['flat_type'],
        'is_apartment': data['is_apartment'],
        'is_penthause': data['is_penthause'],
        'total_area': data['total_area'],
        'living_area': data['living_area'],
        'kitchen_area': data['kitchen_area'],
        'all_rooms_area': data['all_rooms_area'],
        'combined_wcs_count': data['combined_wcs_count'],
        'repair_type': data['repair_type'],
        'floor': data['floor'],
        'price': data['price'],
        'currency': data['currency'],
        'material_type': data['material_type'],
        'floors_count': data['floors_count'],
        'build_year': data['build_year'],
        'ceiling_height': data['ceiling_height'],
        'house_material_type': data['house_material_type'],
        'edit_time': data['edit_time'],
        'publication_date': data['publication_date'],
        'house_material_bti': data['house_material_bti'],
        'house_heating_supply': data['house_heating_supply'],
        'is_emergency': data['is_emergency'],
        'house_overlap_type': data['house_overlap_type'],
        'metro': data['metro'],
        'metro_time': data['metro_time'],
        'travel_type': data['travel_type']
    })


async def process_urls(session: CloudflareScraper, urls: list[str], writer: csv.DictWriter, semaphore: asyncio.Semaphore) -> None:
    """
    Process a list of URLs asynchronously.

    Args:
        session (CloudflareScraper): Async HTTP session.
        urls (list[str]): List of URLs to process.
        writer (csv.DictWriter): CSV writer to write the data.
        semaphore (asyncio.Semaphore): Semaphore for controlling concurrent access.
    """
    tasks = [fetch_data(session, url, writer, semaphore) for url in urls]
    await asyncio.gather(*tasks)


async def main(urls: list[str]) -> None:
    """
    Main function to fetch data from URLs asynchronously and write them to a CSV file.

    Args:
        urls (list[str]): List of URLs to process.
    """
    async with CloudflareScraper() as session:
        filename = 'day2_output.csv'
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'offer_type', 'city', 'okrug', 'raion', 'street', 'house', 'room_count', 'room_type', 'loggiasCount', 'latitude', 'longitude',
                'id', 'phone', 'flat_type', 'is_apartment', 'is_penthause', 'total_area',
                'living_area', 'kitchen_area', 'all_rooms_area', 'combined_wcs_count', 'repair_type',
                'floor', 'price', 'currency', 'material_type', 'floors_count', 'build_year',
                'ceiling_height', 'house_material_type', 'edit_time', 'publication_date',
                'house_material_bti', 'house_heating_supply', 'is_emergency', 'house_overlap_type',
                'metro', 'metro_time', 'travel_type'
            ]  # Adjust fieldnames based on your extracted data
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            semaphore = asyncio.Semaphore(1)
            await process_urls(session, urls, writer, semaphore)

if __name__ == "__main__":
    with open('day2_urls.txt', 'r', encoding='utf-8') as f:
        urls = f.read().split('\n')
    asyncio.run(main(urls))
