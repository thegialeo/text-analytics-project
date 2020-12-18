import os
from os.path import exists, dirname, abspath, join
import requests



def download_TextComplexityDE19():
    """Download the TextComplexityDE dataset from Github Repository.
       Contact Badak Naderi (babak.naderi[at]tu-berlin.de) for further support concerning the dataset itself. 
    """
    # url of dataset github repository
    url = "https://github.com/babaknaderi/TextComplexityDE.git"

    # create folder for dataset (if it doesn't exist yet)
    download_to_path = join(dirname(dirname(abspath(__file__))), "data")
    if not exists(download_to_path):
        os.makedirs(download_to_path)


download_TextComplexityDE19()
