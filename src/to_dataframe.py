import pandas as pd
import os
from os.path import join, abspath, dirname
import textstat

def text_comp19_to_df():

    """
    Returns a pandas Dataframe object with
    the data of the TextComplexityDE19 dataset
    """

    # Path to relevant csv file
    csv_path = join(dirname(dirname(abspath(__file__))),
                    "data","TextComplexityDE19/ratings.csv")

    # read in csv file
    print("Reading in TextComplexityDE19/ratings.csv")
    corpus = pd.read_csv(csv_path, encoding='windows-1252')

    #Rename columns and insert source of this dataframe for consistency
    corpus = corpus.rename(
        columns={"Sentence": "raw_text", "Votes_Complexity": "rating"})

    corpus.insert(2, "source", "text_comp19")

    #Delete all columns except the raw_text and the rating column
    corpus = corpus.drop(columns=
                         ["ID", "Article_ID",
                          "Article", "MOS_Complexity", "Std_Complexity",
                          "Votes_Understandability", "MOS_Understandability",
                          "Std_Understandability", "Vote_Lexical_difficulty",
                          "MOS_Lexical_difficulty", "Std_Lexical_difficulty"
                          ])

    return corpus

def weebit_to_df():

    """
    Returns a pandas Dataframe object with
    the translated data (from english to german)
    of the Weebit dataset.
    """

    # List paths of all .txt files
    print("Reading in Weebit Ele-Txt, Int-Txt, Adv-Txt")
    elementary_path = join(dirname(dirname(abspath(__file__))),
                           "data","WeebitDataset","Texts-SeparatedByReadingLevel","Ele-Txt")
    advanced_path = join(dirname(dirname(abspath(__file__))),
                           "data","WeebitDataset","Texts-SeparatedByReadingLevel","Adv-Txt")
    intermediate_path = join(dirname(dirname(abspath(__file__))),
                           "data","WeebitDataset","Texts-SeparatedByReadingLevel","Int-Txt")

    path_list = [elementary_path, advanced_path, intermediate_path]

    # create dictionary for creation of dataframe
    data_dict = {"raw_text": [], "rating": [], "source": []}

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
                        data_dict["rating"].append(line)
                        continue
                    str_list.append(line)

            # flatten list of line to one big string
            text = ""
            for i in range(0, len(str_list)):
                text += str_list[i]

            # create dataframe out of dictionary
            data_dict["raw_text"].append(text)
            data_dict["source"].append("Weebit")

    weebit_data = pd.DataFrame(data_dict)

    #translate weebit dataset to german

    return weebit_data

def dw_to_df():

    """"
    Returns a pandas Dataframe object with
    the data of the dw dataset
    """

    #.h5 file path
    h5_path = join(dirname(dirname(abspath(__file__))),
                           "data","dw.h5")

    #read in h5 file
    print("Reading in dw.h5")
    data = pd.HDFStore(h5_path)

    #assign in h5 file contained dataframes to variables
    pages_df = data["pages_df"]
    paragraphs_df = data["paragraphs_df"]
    text_df = data["text_df"]

    #merge dataframes on url and append paragraphs and text to one dataframe
    merged = text_df.merge(pages_df, left_on='url'
                           , right_on='url')
    merged2 = paragraphs_df.merge(pages_df, left_on='url',
                                  right_on='url')
    joined = merged.append(merged2, ignore_index=True)

    # Rename, delete columns and insert source of this dataframe for consistency
    dw_set = joined.drop(columns=
                         ["artikel_x", "rubrik_x", "title",
                          "url", "y_x", "rubrik_y", "html",
                          "artikel_y", "tags", "y_y"],
                         )
    dw_set.rename(columns={"text": "raw_text", "levels": "rating"}, inplace=True)
    dw_set.insert(2, "source", "dw")

    return dw_set

def all_data():

    """
    returns one dataframe for all datasets. The datasets are also
    cleared of "\n" and other special symbols.
    """

    # load all datasets into dataframes and store them in variables
    text_comp19 = text_comp19_to_df()
    weebit = weebit_to_df()
    dw = dw_to_df()

    # append all dataframes to one dataframe
    all_dataset = text_comp19.append(weebit, ignore_index=True)
    all_dataset = all_dataset.append(dw, ignore_index=True)

    # delete "\n" and other special symbols
    all_dataset.replace("\n", "", regex=True, inplace=True)

    #add word count to data
    all_dataset['word_count'] = all_dataset['raw_text'].str.findall(r'(\w+)').str.len()

    #add flesch readability index to data
    all_dataset['flesch_readablty'] = all_dataset['raw_text'].apply(textstat.flesch_reading_ease)

    return all_dataset

if __name__ == "__main__":
    #check if data has been downloaded, if not download it.
    pass
    #



