import pandas as pd
import os

def text_comp19_to_df():

    """
    Returns a pandas Dataframe object with
    the data of the TextComplexityDE19 dataset
    """

    # Path to relevant csv file
    csv_path = "..data/TextComplexityDE19/parallel_corpus.csv"

    # read in csv file
    parallel_corpus = pd.read_csv(csv_path)

    return parallel_corpus

def weebit_to_df():

    """
    Returns a pandas Dataframe object with
    the data of the Weebit dataset
    """

    # List paths of all .txt files
    elementary_path = \
        "../data/WeebitDataset/Texts-SeparatedByReadingLevel/Ele-Txt"
    advanced_path = \
        "../data/WeebitDataset/Texts-SeparatedByReadingLevel/Adv-Txt"
    intermediate_path = \
        "../data/WeebitDataset/Texts-SeparatedByReadingLevel/Int-Txt"

    path_list = [elementary_path, advanced_path, intermediate_path]

    # create dictionary for creation of dataframe
    data_dict = {"raw_text": [], "difficulty": [], "origin": []}

    # Read in .txt files and write to dataframe
    for path in path_list:
        for filename in os.listdir(path):
            full_path_to_file = os.path.join(path, filename)

            # omit first line of each file, it contains the difficulty of the file
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

def dw_to_df():

    """"
    Returns a pandas Dataframe object with
    the data of the dw dataset
    """

    #.h5 file path
    h5_path = "../data/dw.h5"

    #read in h5 file
    data = pd.HDFStore(h5_path)

    #assign in h5 file contained dataframes to variables
    pages_df = data["pages_df"]
    paragraphs_df = data["paragraphs_df"]
    text_df = data["text_df"]

    #merge dataframes on url and join paragraphs and text to one dataframe
    merged = text_df.merge(pages_df, left_on='url'
                           , right_on='url')
    merged2 = paragraphs_df.merge(pages_df, left_on='url',
                                  right_on='url')
    joined = merged.append(merged2, ignore_index=True)

    return joined

if __name__ == "__main__":
    pass



