import pandas as pd
import os


def write_to_df_weebit():
    """
    Write text from .txt files into one dataframe file
    """

    # List paths of all .txt files
    elementary_path = "../data/WeebitDataset/Texts-SeparatedByReadingLevel/Ele-Txt"
    advanced_path = "../data/WeebitDataset/Texts-SeparatedByReadingLevel/Adv-Txt"
    intermediate_path = "../data/WeebitDataset/Texts-SeparatedByReadingLevel/Int-Txt"

    path_list = [elementary_path, advanced_path, intermediate_path]

    # create dictionary for creation of dataframe
    data_dict = {"raw_text": [], "difficulty": [], "origin": []}

    # Read in .txt files and write to dataframe
    for path in path_list:
        for filename in os.listdir(path):
            full_path_to_file = os.path.join(path, filename)

            # omit first line of each file, because it contains the difficulty of the file
            omit = True
            str_list = []

            with open(full_path_to_file, encoding='windows-1252') as file:
                for line in file:
                    if omit:
                        omit = False
                        # write difficulty to dataframe
                        data_dict["difficulty"].append(line)
                        continue
                    str_list.append(line)

            # flatten list of line to one big string
            text = ""
            for i in range(0, len(str_list)):
                text += str_list[i]

            # create dataframe out of dictionary
            data_dict["raw_text"].append(text)
            data_dict["origin"].append("Weebit")

    weebit_data = pd.DataFrame(data_dict)

    return weebit_data





