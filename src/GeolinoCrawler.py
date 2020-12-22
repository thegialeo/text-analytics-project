#!/usr/bin/env python
# coding: utf-8

# ## Früher Version des Crawlers für www.geo.de/geolino und www.geo.de

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv

def get_paragraphs(url):
    text1 = []
    r = requests.get(url)
    doc = BeautifulSoup(r.text, "html.parser")
    for elt in doc.select(".paragraph"):
        print(elt)
        text1.append(elt.text)
    if text1 == []:
        pass
    
    else:
        return text1

def expand_menu_urls(url):
    urls = []
    r = requests.get(url)
    doc = BeautifulSoup(r.text, "html.parser")
    for elt in doc.select(".expanded"):
        try:
            joint = urljoin(url,elt.attrs["href"])
            if joint != url and joint not in urls:
                urls.append(joint)
        except KeyError:
            pass
    
    return(urls)
    
def get_urls(url):
    urls = []
    r = requests.get(url)
    doc = BeautifulSoup(r.text, "html.parser")
    for elt in doc.select("a"):
        try:
            joint = urljoin(url,elt.attrs["href"])
            if joint != url and joint not in urls:
                urls.append(joint)
        except KeyError:
            pass
    
    return(urls)

first_urls = get_urls("https://www.geo.de/geolino")
#second_urls = [get_urls(url) for url in first_urls]

expand_menu_urls("https://www.geo.de/geolino")

urls = []
GeolinoURL = "https://www.geo.de/geolino"
r = requests.get(GeolinoURL)
doc = BeautifulSoup(r.text, "html.parser")
for expanded in doc.select(".expanded"):
    urls.append([x.attrs["href"] for x in expanded.select("a")])

urls2 = []
for liste in urls:
    for elt in liste:
        urls2.append(elt)

urls2 = [urljoin(GeolinoURL,url) for url in urls2] #join url und geolino url

print(urls2)



