import os
import shutil
from distutils.dir_util import copy_tree
from os.path import abspath, dirname, exists, join


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

    # check if data folder already has subfolder TextComplexityDE, if no: download github repository to folder
    if "TextComplexityDE19" not in os.listdir(download_to_path):
        # create temp folder to download github repository
        if not exists(join(download_to_path, "temp")):
            os.makedirs(join(download_to_path, "temp"))
        os.system("git clone {} {}".format(url, join(download_to_path, "temp")))
        # clean up downloaded repository
        shutil.rmtree(join(download_to_path, "temp", ".git"))
        os.remove(join(download_to_path, "temp", ".gitignore"))
        os.remove(join(download_to_path, "temp", "LICENSE"))
        os.remove(join(download_to_path, "temp", "README.md"))
        # copy content from temp folder to data folder
        copy_tree(join(download_to_path, "temp"), download_to_path)
        # delete temp folder
        shutil.rmtree(join(download_to_path, "temp"))
    else:
        print("Folder data already contains TextComplexityDE19. Please delete the folder TextComplexityDE and remove from trash. Rerun the code.")
        exit()


if __name__ == "__main__":
    download_TextComplexityDE19()
