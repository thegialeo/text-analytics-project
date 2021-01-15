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

    #Rename columns and insert source of this dataframe for consistency
    parallel_corpus = parallel_corpus.rename(
        columns={"Original_Sentence": "raw_text", "Rating": "rating"})

    parallel_corpus.insert(2, "source", "text_comp19")

    #Delete all columns except the raw_text and the rating column
    parallel_corpus = parallel_corpus.drop(columns=
                                           ["Sentence_Id", "Article_ID",
                                            "Article", "Simplification"])

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
    returns one dataframe for all datasets
    """

    # load all datasets into dataframes and store them in variables
    text_comp19 = text_comp19_to_df()
    weebit = weebit_to_df()
    dw = dw_to_df()

    # append all dataframes to one dataframe
    all_dataset = text_comp19.append(weebit, ignore_index=True)
    all_dataset = all_dataset.append(dw, ignore_index=True)

    return all_dataset

if __name__ == "__main__":
    pass



