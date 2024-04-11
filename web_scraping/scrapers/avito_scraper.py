import cloudscraper
from bs4 import BeautifulSoup




scraper = cloudscraper.create_scraper()
link = 'https://www.avito.ru/all/nedvizhimost'
response = scraper.get(url=link)
soup = BeautifulSoup(response.text)
print(soup)