from os.path import abspath, dirname, join

import nltk
import spacy
import stop_words


def get_stopwords(source="nltk"):
    """Return German stopwords based on given source.

    Args:
        source (str, optional): source to load stopwords from (options: "spacy", "nltk", "stop_words", "german_plain", "german_full"). Defaults to "nltk".

    Return:
        stopwords (list): a list of german stopwords
    """

    if source == "nltk":
        return nltk.corpus.stopwords.words('german')
    elif source == "spacy":
        spacy.load("de_core_news_sm", disable=["tagger", "parser","ner"])
        return list(spacy.lang.de.stop_words.STOP_WORDS)
    elif source == "stop_words":
        return stop_words.get_stop_words('de')
    elif source == "german_plain":
        path = join(dirname(dirname(dirname(abspath(__file__)))), "corpus", "stopwords", "german_stopwords_plain.txt")
        return [line.rstrip('\n') for line in open(path)][9:]
    elif source == "german_full":
        path = join(dirname(dirname(dirname(abspath(__file__)))), "corpus", "stopwords", "german_stopwords_full.txt")
        return [line.rstrip('\n') for line in open(path)][9:]
    else:
        print("stopword source {} is not implemented. Please select one the folling options: 'spacy', 'nltk', 'stop_words', 'german_plain', 'german_full'".format(source))
        exit()
        