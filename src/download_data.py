import os
from os.path import exists, dirname, abspath, join
import shutil
import requests


def download_TextComplexityDE19():
    """Download the TextComplexityDE dataset from Github Repository.
    Contact Badak Naderi (babak.naderi[at]tu-berlin.de) for further support concerning the dataset itself.

    Written by Leo Nguyen. Contact Xenovortex, if problems arises.
    """
    # url of dataset github repository
    url = "https://github.com/babaknaderi/TextComplexityDE.git"

    # create folder for dataset (if it doesn't exist yet)
    download_to_path = join(dirname(dirname(abspath(__file__))), "data")
    if not exists(download_to_path):
        os.makedirs(download_to_path)

    # check if data folder empty, if yes: download github repository to folder
    if len(os.listdir(download_to_path)) == 0:
        os.system("git clone {} {}".format(url, download_to_path))
        # clean up downloaded repository
        shutil.rmtree(join(download_to_path, ".git"))
        os.remove(join(download_to_path, ".gitignore"))
        os.remove(join(download_to_path, "LICENSE"))
        os.remove(join(download_to_path, "README.md"))
    else:
        print("Folder data is not empty. Please delete the folder data and remove from trash. Rerun the code.")
        exit()


if __name__ == "__main__":
    download_TextComplexityDE19()
