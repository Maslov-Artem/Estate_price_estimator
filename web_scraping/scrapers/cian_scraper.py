import asyncio
from datetime import datetime
from aiocfscrape import CloudflareScraper
from bs4 import BeautifulSoup
import re
from math import ceil
from fake_headers import Headers


def get_headers() -> dict[str, str]:

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


URL = 'https://cian.ru/cat.php?'
DEFAULT_PATH = 'currency=2&deal_type=sale&engine_version=2'

price_dict = {
    1000000: 5000000,
    5000000: 6000000,
    6000000: 6500000,
    6500000: 7000000,
    7000000: 7300000,
    7300000: 7500000,
    7500000: 7700000,
    7700000: 7900000,
    7900000: 8100000,
    8100000: 8300000,
    8300000: 8500000,
    8500000: 8700000,
    8700000: 8900000,
    8900000: 9050000,
    9050000: 9250000,
    9250000: 9450000,
    9450000: 9550000,
    9550000: 9750000,
    9750000: 9900000,
    9900000: 10000000,
    10000000: 10160000,
    10160000: 10300000,
    10300000: 10450000,
    10450000: 10550000,
    10550000: 10700000,
    10700000: 10800000,
    10800000: 10900000,
    10900000: 10990000,
    10990000: 11100000,
    11100000: 11250000,
    11250000: 11400000,
    11400000: 11500000,
    11500000: 11600000,
    11600000: 11750000,
    11750000: 11900000,
    11900000: 11990000,
    11990000: 12090000,
    12090000: 12250000,
    12250000: 12450000,
    12450000: 12550000,
    12550000: 12750000,
    12750000: 12950000,
    12950000: 13050000,
    13050000: 13250000,
    13250000: 13450000,
    13450000: 13600000,
    13600000: 13800000,
    15700000: 15950000,
    15950000: 16200000,
    16200000: 16450000,
    16450000: 16750000,
    16750000: 17050000,
    17050000: 17450000,
    17450000: 17750000,
    17750000: 18050000,
    18050000: 18450000,
    18450000: 18850000,
    18850000: 19300000,
    19300000: 19700000,
    19700000: 20000000,
    20000000: 20450000,
    20450000: 20950000,
    20950000: 21450000,
    21450000: 21950000,
    21950000: 22450000,
    22450000: 22950000,
    22950000: 23600000,
    23600000: 24500000,
    24500000: 25200000,
    25200000: 26200000,
    26200000: 27400000,
    27400000: 28800000,
    28800000: 30000000,
    30000000: 31500000,
    31500000: 33200000,
    33200000: 35500000,
    35500000: 39000000,
    39000000: 44000000,
    44000000: 54000000,
    54000000: 79000000,
    79000000: 999999999
}


# Function to scrape URLs asynchronously
async def scrape_urls_async(session: CloudflareScraper,
                            url: str,
                            headers: Dict[str, str],
                            proxy: str,
                            min_price: int,
                            max_price: int,
                            retry_limit: int = 3) -> List[str]:
    """
    Scrape cian URLs asynchronously and return a list of addvertisment URLs.

    Args:
        session (CloudflareScraper): Async HTTP session.
        url (str): URL to scrape.
        headers (dict): Request headers.
        proxy (str): Proxy server URL.
        min_price (int): Minimum price filter.
        max_price (int): Maximum price filter.
        retry_limit (int): Number of retry attempts.

    Returns:
        list: List of scraped URLs.
    """
    urls = []
    try:
        retry_count = 0
        while retry_count < retry_limit:
            try:
                async with session.get(url, headers=headers, proxy=proxy) as response:
                    if response.status == 200:
                        html = await response.text()
                        bs = BeautifulSoup(html, 'html.parser')
                        url_tags = bs.find_all(
                            'div', {'data-name': 'GeneralInfoSectionRowComponent'})
                        for tag in url_tags:
                            url_tag = tag.find('a')
                            if url_tag:
                                try:
                                    link = url_tag['href']
                                    if 'www.cian.ru/sale' in link:
                                        urls.append(link)
                                except KeyError:
                                    print('Abort mission, no links found')
                        n_adds = bs.find_all(
                            'div', {'data-name': 'SummaryHeader'})[0].find('h5').text
                        n_pages = ceil(int(re.sub(r'\D', '', n_adds))/28) + 1
                        if n_pages > 54:
                            n_pages = 54
                        print(n_pages)
                        for i in range(2, n_pages):
                            url = f'{URL}{DEFAULT_PATH}&maxprice={max_price}&minprice={min_price}&offer_type=flat&p={i}&region=1&room1=1&room2=1&room9=1'
                            async with session.get(url, headers=headers, proxy=proxy) as response:
                                if response.status == 200:
                                    html = await response.text()
                                    bs = BeautifulSoup(html, 'html.parser')
                                    url_tags = bs.find_all(
                                        'div', {'data-name': 'GeneralInfoSectionRowComponent'})
                                    for tag in url_tags:
                                        url_tag = tag.find('a')
                                        if url_tag:
                                            try:
                                                link = url_tag['href']
                                                if 'www.cian.ru/sale' in link:
                                                    urls.append(link)
                                            except KeyError:
                                                print(
                                                    'Abort mission, no links found')
                        break
            except Exception as e:
                print(f'Retrying due to error: {e}')
                retry_count += 1
    except Exception as e:
        print('Error:', e)
    return urls


async def main():
    """
    Main function to scrape URLs asynchronously and save them to a file.
    """
    urls = []
    proxy = 'http://d5933ca885c2e541:RNW78Fm5@res.proxy-seller.io:10000'
    start_time = datetime.now()
    async with CloudflareScraper() as session:
        tasks = []
        for _, (min_price, max_price) in enumerate(price_dict.items()):
            url = f"{URL}{DEFAULT_PATH}&maxprice={max_price}&minprice={min_price}&offer_type=flat&region=1&room1=1&room2=1&room9=1"
            headers = get_headers()
            task = asyncio.create_task(scrape_urls_async(
                session, url, headers, proxy, min_price, max_price))
            tasks.append(task)
        urls = await asyncio.gather(*tasks)
    print(f'Execution time = {datetime.now() - start_time}')
    with open('urls.txt', 'w') as f:
        for url_list in urls:
            for url in url_list:
                f.write(url + '\n')


if __name__ == '__main__':
    asyncio.run(main())
