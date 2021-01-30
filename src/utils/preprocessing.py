import spacy
import nltk
import stop_words


def get_stopwords(source="nltk"):
    """Return German stopwords based on given source.

    Args:
        source (str, optional): source to load stopwords from (options: "spacy", "nltk", "german_plain", "german_full"). Defaults to "nltk".

    Return:
        stopwords (list): a list of german stopwords
    """

    if source == "nltk":
        return nltk.corpus.stopwords.words('german')
    elif source == "spacy":
        spacy.load("de_core_news_sm", disable=["tagger", "parser","ner"])
        return list(spacy.lang.de.stop_words.STOP_WORDS)
    elif source == "stop-words":
        return stop_words.get_stop_words('de')
    elif source == "german_plain":
        pass
        