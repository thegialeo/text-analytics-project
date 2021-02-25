import re
from os import path

import normalization
import numpy as np
import pandas as pd
import spacy

# import to_dataframe
import wordlists


def construct_features(sentence, verbose=False):
    """constructs a #sentences × #features numpy array, rows are sentences, columns
    are features. use by passing a dataframe column containing (normalized) sentences.

    Kwargs:
    sentence -- a dataframe column containing normalized sentences.
    """
    my_df = pd.DataFrame()

    # ======= Counting Commas ========
    my_df["commas"] = count_commas(sentence)
    sentence = normalization.normalize_sentence(sentence)
    # sadly, counting commas only works when they haven't already been removed
    if my_df["commas"].isnull().all():
        my_df.drop(columns="commas")

    my_df[["words", "letters"]] = count_words_and_letters(sentence)
    my_df["syllables"] = sentence.str.split().apply(count_syllables)
    my_df["monosyllables"] = sentence.apply(count_monosyllables)
    my_df["ge3syllables"] = sentence.apply(count_polysyllables, args=(3,))
    my_df["long_words"] = sentence.apply(count_long_words, args=(6,))
    my_df["infrequent100"] = sentence.apply(count_infrequent_words, args=(100,))
    my_df["infrequent1000"] = sentence.apply(count_infrequent_words, args=(1000,))
    my_df["wstf"] = wiener_sachtextformel(
        my_df["ge3syllables"],
        my_df["words"],
        my_df["long_words"],
        my_df["monosyllables"],
    )

    # ======= POS tag density =======
    my_df[
        [
            "nouns",
            "propernouns",
            "pronouns",
            "conj",
            "adj",
            "adv",
            "ver",
            "aux",
            "not_pron_or_det",
        ]
    ] = POS_tag_density(sentence)

    if verbose:
        print(
            "==============\nFrom ",
            len(sentence),
            " sentences, constructing size ",
            my_df.shape,
            " feature dataframe. \nThe feature names are: ",
            my_df.columns,
            "\n==============",
            sep="",
        )

    return my_df


def count_commas(sentence):
    return sentence.str.count(",")


def count_words_and_letters(sentence):
    """Counts the basic statistics number of words and number of letters. Takes a
    dataframe column of sentences as the input and returns a new dataframe with two
    columns for word_count and letter_count.
    letter_count only counts so-called word symbols, so letters and numbers, not
    punctuation. word_count splits the sentence at whitespaces and gives the number
    of components after the split.

    Keyword arguments:
    sentence -- a dataframe column containing sentences, maybe also similar structures
    """
    word_count = sentence.str.split().str.len()
    letter_count = sentence.str.count(r"\w")
    return pd.DataFrame({"word_count": word_count, "letter_count": letter_count})


def count_syllables(words):
    """Given a list of words, counts the syllables in each and returns the sum.
    Counting syllables doesn't necessarily give very good results yet

    TODO improve syllable counting method

    Keyword arguments:
    words -- list of words (use split() on sentences, maybe)
    """
    cc_pattern = re.compile("[^aeiouyäöü]{2,}")  # two consonants in a row
    sentence_syllables = 0
    for word in words:
        word_syllables = 1
        current_pos = len(word) - 1
        while current_pos >= 0:
            current_character = word[current_pos]
            current_pos -= 1
            if current_character in "aeiouyäöü":
                if current_pos <= 0:
                    break
                else:
                    current_character = word[current_pos]
                    if current_character not in "aeiouyäöü":
                        word_syllables += 1
                    current_pos -= 1
        if cc_pattern.match(word) and len(word) > 2:
            word_syllables -= 1
        sentence_syllables += word_syllables
    return sentence_syllables


def count_polysyllables(sentence, threshold=2):
    """Given a sentence, computes and returns the number of polysyllabic words
    containing at least a certain, optionally specified amount of syllables

    Keyword arguments:
    sentence -- sentence or series of sentences
    threshold -- (optional) this function will count words with at least this many
    syllables (default 2)
    """
    cc_pattern = re.compile("[^aeiouyäöü]{2,}")
    polysyllables = 0
    words = sentence.split()
    for word in words:
        word_syllables = 1
        current_pos = len(word) - 1
        while current_pos >= 0:
            current_character = word[current_pos]
            current_pos -= 1
            if current_character in "aeiouyäöü":
                if current_pos <= 0:
                    break
                else:
                    current_character = word[current_pos]
                    if current_character not in "aeiouyäöü":
                        word_syllables += 1
                    current_pos -= 1
        if cc_pattern.match(word) and len(word) > 2:
            word_syllables -= 1
        if word_syllables >= threshold:
            polysyllables += 1
    return polysyllables


def count_monosyllables(sentence):
    """Given a sentence, computes and returns the number of monosyllabic words
    (words which are just 1 syllable long) contained within.

    Keyword arguments:
    sentence -- sentence or series of sentences
    """
    cc_pattern = re.compile("[^aeiouyäöü]{2,}")
    monosyllables = 0
    words = sentence.split()
    for word in words:
        word_syllables = 1
        current_pos = len(word) - 1
        while current_pos >= 0:
            current_character = word[current_pos]
            current_pos -= 1
            if current_character in "aeiouyäöü":
                if current_pos <= 0:
                    break
                else:
                    current_character = word[current_pos]
                    if current_character not in "aeiouyäöü":
                        word_syllables += 1
                    current_pos -= 1
        if cc_pattern.match(word) and len(word) > 2:
            word_syllables -= 1
        if word_syllables == 1:
            monosyllables += 1
    return monosyllables


def count_infrequent_words(sentence, size=100):
    if size == 100:
        wordlist = wordlists.uni_leipzig_top100de()
    elif size == 1000:
        wordlist = wordlists.uni_leipzig_top1000de()
    else:
        print(
            "count_infreduent_words was called with an unsupported wordlist size. (Implemented so far: 100, 1000)"
        )
        return None
    number_of_words = 0
    for word in sentence.split():
        if word not in wordlist:
            number_of_words += 1
    return number_of_words


def count_long_words(sentence, length):
    """Given a sentence, computes and returns the number of words equal to or longer
    than the provided threshold length

    Keyword arguments:
    sentence -- sentence or series of sentences
    length -- minimum length for a word to be counted
    """
    long_words = 0
    words = sentence.split()
    for word in words:
        if len(word) >= length:
            long_words += 1
    return long_words


def wiener_sachtextformel(
    polysyllables_count, word_count, long_words_count, monosyllables_count
):
    """Computes the first wiener sachtextformel, using the number of words with 3 or
    more syllables, the number of words, the number of words with 6 or more
    letters and the number of monosyllabic words

    Keyword arguments:
    polysyllables_count -- number of words with three or more syllables
    word_count -- number of words
    long_words_count -- number of words with 6 or more letters
    monosyllables_count -- number of words with only a single syllable
    """

    return (
        0.1935 * polysyllables_count / word_count
        + 0.1672 * word_count
        + 0.1297 * long_words_count / word_count
        - 0.0327 * monosyllables_count / word_count
        - 0.875
    )


def POS_tag_density(sentences):
    nlp = spacy.load("de_core_news_sm")

    nouns_list = []
    propernouns_list = []
    pronouns_list = []
    conj_list = []
    adj_list = []
    adv_list = []
    ver_list = []
    aux_list = []
    not_pron_or_det_list = []

    for sentence in sentences:
        doc = nlp(sentence)

        nouns = 0
        propernouns = 0
        pronouns = 0
        conj = 0
        adj = 0
        adv = 0
        ver = 0
        aux = 0
        not_pron_or_det = len(doc)

        for token in doc:
            if token.pos_ == "NOUN":
                nouns += 1
            elif token.pos_ == "PROPN":
                propernouns += 1
            elif token.pos_ == "PRON":
                pronouns += 1
                not_pron_or_det -= 1
            elif token.pos_ == "SCONJ" or token.pos_ == "CCONJ":
                conj += 1
            elif token.pos_ == "ADJ":
                adj += 1
            elif token.pos_ == "ADV":
                adv += 1
            elif token.pos_ == "VERB":
                ver += 1
            elif token.pos_ == "AUX":
                aux += 1
            elif token.pos_ == "DET":
                not_pron_or_det -= 1

        nouns = nouns * 1.0 / len(doc)
        propernouns = propernouns * 1.0 / len(doc)
        pronouns = pronouns * 1.0 / len(doc)
        conj = conj * 1.0 / len(doc)
        adj = adj * 1.0 / len(doc)
        adv = adv * 1.0 / len(doc)
        ver = ver * 1.0 / len(doc)
        aux = aux * 1.0 / len(doc)

        nouns_list.append(nouns)
        propernouns_list.append(propernouns)
        pronouns_list.append(pronouns)
        conj_list.append(conj)
        adj_list.append(adj)
        adv_list.append(adv)
        ver_list.append(ver)
        aux_list.append(aux)
        not_pron_or_det_list.append(not_pron_or_det)

    return pd.DataFrame(
        {
            "nouns": nouns_list,
            "propernouns": propernouns_list,
            "pronouns": pronouns_list,
            "conj": conj_list,
            "adj": adj_list,
            "adv": adv_list,
            "ver": ver_list,
            "aux": aux_list,
            "not_pron_or_det": not_pron_or_det_list,
        }
    )


if __name__ == "__main__":
    # load TextComplexityDE dataset
    # df_all = to_dataframe.text_comp19_to_df()
    # df_all.columns = df_all.columns.str.lower()
    # df_all.head()
    df_all = pd.read_excel(
        path.join(
            path.dirname(path.abspath(__file__)),
            "..",
            "data",
            "TextComplexityDE19.xlsx",
        ),
        engine="openpyxl",
        sheet_name=2,
        header=1,
    )
    df_all.columns = df_all.columns.str.lower()
    print(df_all["sentence"])

    feature_matrix = construct_features(df_all["sentence"], verbose=True)
    print(feature_matrix)
    print("\n", feature_matrix.iloc[0, :])

    nlp = spacy.load("de_core_news_sm")
    doc = df_all["sentence"][0]
    print(doc)
    doc = nlp(doc)

    print(len(doc))
    for token in doc:
        print(
            token.text,
            token.lemma_,
            token.pos_,
            token.tag_,
            "=",
            spacy.explain(token.tag_),
            "=======",
            token.dep_,
            token.shape_,
            token.is_alpha,
            token.is_stop,
            # token.morph,
        )

    print("Explain ADP:", spacy.explain("ADP"))
    print("Explain DET:", spacy.explain("DET"))
    print("Explain PRON:", spacy.explain("PRON"))
    print("Explain SCONJ:", spacy.explain("SCONJ"))
    print("Explain CCONJ:", spacy.explain("CCONJ"))
    print("Explain VERB:", spacy.explain("VERB"))
    print("Explain AUX:", spacy.explain("AUX"))
    print("Explain INTJ:", spacy.explain("INTJ"))

    # for chunk in doc.noun_chunks:
    #    print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
