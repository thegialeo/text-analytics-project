import pandas as pd
import os
from os.path import join, abspath, dirname
import textstat
from google_trans_new import google_translator
#import exploration
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
import spacy
from nltk.stem import SnowballStemmer

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
        columns={"Sentence": "raw_text", "MOS_Complexity": "rating"})

    corpus.insert(2, "source", "text_comp19")

    #Delete all columns except the raw_text and the rating column
    corpus = corpus.drop(columns=
                         ["ID", "Article_ID",
                          "Article", "Votes_Complexity", "Std_Complexity",
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
    elementary_path = join(dirname(dirname(dirname(abspath(__file__)))),
                           "data","WeebitDataset","Texts-SeparatedByReadingLevel","Ele-Txt")
    advanced_path = join(dirname(dirname(dirname(abspath(__file__)))),
                           "data","WeebitDataset","Texts-SeparatedByReadingLevel","Adv-Txt")
    intermediate_path = join(dirname(dirname(dirname(abspath(__file__)))),
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
    print("Translating Weebit dataset to german...")
    trans = google_translator()
    weebit_data["raw_text"] = weebit_data["raw_text"].\
        apply(lambda x: trans.translate(x, lang_tgt="de"))

    return weebit_data

def dw_to_df():

    """"
    Returns a pandas Dataframe object with
    the data of the dw dataset
    """

    #.h5 file path
    h5_path = join(dirname(dirname(dirname(abspath(__file__)))),
                           "data", "dw", "dw.h5")

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
    cleared of "\n" and other special symbols, numbers, whitespace sequences.
    Also word count and flesch readability index is added to data.
    """

    # load all datasets into dataframes and store them in variables
    text_comp19 = text_comp19_to_df()
    #weebit = weebit_to_df()
    dw = dw_to_df()

    # append all dataframes to one dataframe
    all_dataset = text_comp19.append(dw, ignore_index = True)
    #all_dataset = all_dataset.append(dw, ignore_index = True)

    # delete "\n" and other special symbols
    print("removing newline command")
    all_dataset.replace("\n", "", regex=True, inplace = True)

    # remove numbers from data
    print("removing numbers from data")
    all_dataset.raw_text.replace(r"\d", "", regex=True, inplace = True)

    # remove punctuation from data
    print("removing punctuation from data")
    all_dataset["raw_text"] = all_dataset["raw_text"].apply(lambda x: exploration.remove_punctuation(x))

    # remove whitespace from data
    print("removing whitespace sequences from data")
    all_dataset["raw_text"] = all_dataset["raw_text"].apply(lambda x: exploration.remove_whitespace(x))

    #add word count to data
    print("adding word count to data")
    all_dataset['word_count'] = all_dataset['raw_text'].str.findall(r'(\w+)').str.len()

    #add flesch readability index to data
    print("adding flesch readability index to data")
    all_dataset['flesch_readablty'] = all_dataset['raw_text'].apply(textstat.flesch_reading_ease)

    return all_dataset


def augmented_all(backtrans = False, lemmatization = False,
                  stemming = False, randword_swap = False,
                  randword_del = False, test_size = 0.1, stopwords = False):

    """
    Returns the augmented training dataset
    and the test dataset of all the data.

    backtrans : enables back and forth translation of the data
    lemmatization self explanatory
    stemming self explanatory
    randword_swap : enables randomly swapping words around sentences
    randword_del : enalbles randomly deleting words from sentences
    test_size : gives the ratio of test to train set

    train_set, test_set = augmented_all()

    """

    # Perform a Train-Test Split keeping dataset proportions the same
    print("perform train-test split keeping dataset proportions the same")
    all_dataset = all_data()
    text_comp_train, text_comp_test = train_test_split(
        all_dataset[all_dataset["source"] == "text_comp19"], test_size=test_size)

    #weebit_train, weebit_test = train_test_split(all_dataset[all_dataset["source"] == "Weebit"],
     #                                            test_size=test_size)

    dw_train, dw_test = train_test_split(all_dataset[all_dataset["source"] == "text_comp19"],
                                         test_size=test_size)

    all_dataset_train = text_comp_train.append(dw_train, ignore_index=True)
    #all_dataset_train = all_dataset_train.append(dw_train, ignore_index=True)

    all_dataset_test = text_comp_test.append(dw_train, ignore_index=True)
    #all_dataset_test = all_dataset_test.append(dw_train, ignore_index=True)

    ## Augmentation of data
    print("Start augmenting Data...")

    # Back and forth translation of data
    if backtrans == True:

        print("Back and forth translation...")
        back_translation_aug = naw.BackTranslationAug(
            from_model_name='transformer.wmt19.de-en',
            to_model_name='transformer.wmt19.en-de')

        translated = all_dataset_train
        translated["raw_text"] = translated["raw_text"] \
            .apply(lambda x: back_translation_aug.augment(x))

        all_dataset_train = all_dataset_train.append(translated, ignore_index=True)

    # Random word swap
    if randword_swap == True:
        print("Random word swap")
        aug1 = naw.RandomWordAug(action="swap")
        swapped_data = all_dataset_train
        swapped_data["raw_text"] = all_dataset_train["raw_text"].apply(lambda x: aug1.augment(x))
        all_dataset_train = all_dataset_train.append(swapped_data, ignore_index=True)

    # Random word deletion
    if randword_del == True:

        print("Random word deletion")
        aug2 = naw.RandomWordAug()
        rand_deleted_data = all_dataset_train
        rand_deleted_data["raw_text"] = all_dataset_train["raw_text"].apply(lambda x: aug2.augment(x))
        all_dataset_train = all_dataset_train.append(rand_deleted_data, ignore_index=True)

    # Remove Stopwords
    if stopwords == True:
        print("removing stopwords...")
        pass

    # Lemmatization using spacy
    if lemmatization == True:

        print("lemmatizing")
        nlp = spacy.load('de_core_news_sm')
        all_dataset_train["raw_text"] = all_dataset_train["raw_text"]\
            .apply(lambda x: ' '.join([y.lemma_ for y in nlp(x)]) )

    # Stemming using
    if stemming == True:

        print("stemming")
        stemmer = SnowballStemmer("german")
        all_dataset_train["raw_text"] = all_dataset_train["raw_text"] \
            .apply(lambda x: stemmer.stem(x))

    return all_dataset_train, all_dataset_test


def store_augmented_h5(filename = "",backtrans = False, lemmatization = False,
                  stemming = False, randword_swap = False,
                  randword_del = False, test_size = 0.1):

    """
    Since the augmented dataset is a large file and
    all the preprocessing steps require a long time
    to finish it is reasonable to save this data once completed.

    Args:
    backtrans : enables back and forth translation of the data
    lemmatization : self explanatory
    stemming : self explanatory
    randword_swap : enables randomly swapping words around sentences
    randword_del : enalbles randomly deleting words from sentences
    test_size : gives the ratio of test to train set

    The file is saved in the same data folder where the original data also resides.
    filename = "filename.h5", keys ="train","test"
    """
    # Ask user for filename
    if filename == "":
        filename = input("Please enter filename with .h5 at the end")

    # define path of .HDF5 file
    h5_path = join(dirname(dirname(dirname(abspath(__file__)))),
                   "data", filename)

    # Load augmented data into variables
    all_dataset_train, all_dataset_test = augmented_all(backtrans, lemmatization,
                  stemming, randword_swap, randword_del, test_size)

    # Write augmented data to h5 file at the above path "h5_path"
    all_dataset_train.to_hdf(h5_path, key="train")
    all_dataset_test.to_hdf(h5_path, key="test")

def read_augmented_h5(filename = ""):
    """
    Args: filename (default empty, but will be prompted to enter name)

    Returns the augmented data from the stored .HDF5 file.
    Similar to augmented_all() with the difference that the
    data is not generated but read instead.

    train_set, test_set = read_augmented_h5().
    """
    # Ask user for
    if filename == "":
        filename = input("Please enter filename with .h5 at the end")

    # define path of .HDF5 file
    h5_path = join(dirname(dirname(dirname(abspath(__file__)))),
                   "data", filename)

    # read in .HDF5 file
    data = pd.HDFStore(h5_path)

    return data["train"], data["test"]

if __name__ == "__main__":
    store_augmented_h5()



