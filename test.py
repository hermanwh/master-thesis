import requests

URL = "http://factpages.npd.no/Default.aspx?culture=nb-no&nav1=field&nav2=TableView%7cProduction%7cSaleable%7cMonthly"
URL = "https://www.ntnu.no/studier/emner/TDT4100"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
    "Accept-Encoding": "*",
    "Connection": "keep-alive"
}


print(URL)

page = requests.get(URL, headers={'Accept-Encoding': None})

pprint(asd)
