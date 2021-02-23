from os.path import abspath, dirname, join

import nltk
import spacy
import stop_words


def get_stopwords(source="spacy"):
    """Return German stopwords based on given source.

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

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
        raise ValueError("stopword source {} is not implemented. Please select one of the following options: 'spacy', 'nltk', 'stop_words', 'german_plain', 'german_full'".format(source))


def tokenizer(df, method='spacy'):
    """Tokenizer that takes a dataframe of sentences and returns a 2d list containing token lists for each sentence.

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        df (pandas dataframe): takes a 1d dataframe of sentences
        method (str, optional): packages to use for tokenization (options: 'nltk', 'spacy'). Defaults to 'nltk' 

    Return:
        corpus (list): 2d python list (list containing list of tokens for each sentence)
    """

    if method == 'nltk':
        data = df.apply(lambda x: str(x).lower()).to_list()
        corpus = [nltk.word_tokenize(line, language="german") for line in data]
        return corpus
    elif method == 'spacy':
        data = df.apply(lambda x: str(x).lower()).to_list()
        nlp = spacy.load("de_core_news_sm", disable=["tagger", "parser","ner"])
        corpus = [[token.text for token in nlp(line)] for line in data]
        return corpus
    else:
        raise ValueError("method {} is not implemented. Please select one of following options: 'ntlk', 'spacy'")


