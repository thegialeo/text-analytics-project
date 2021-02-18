import re

import numpy as np
import pandas as pd
from utils import wordlists


def construct_features(sentence, normalize=False):
    """constructs a #sentences × #features numpy array, rows are sentences, columns
    are features. use by passing a dataframe column containing (normalized) sentences
    and optionally, set normalize to true.

    feature count at time of writing this: 9 features

    Kwargs:
    sentence -- a dataframe column containing normalized sentences.
    normalize (optional) -- normalize feature columns to the same range (default off)
    when normalized, all values are between 0 and 100, otherwise theyre integer counts
    """
    my_df = pd.DataFrame()
    my_df[["words", "letters"]] = count_words_and_letters(sentence)
    my_df["words_not_pronouns_articles"] = (
        my_df["words"]
        - sentence.apply(count_pronouns)
        - sentence.apply(count_definite_articles)
    )
    my_df["syllables"] = sentence.str.split().apply(count_syllables)
    my_df["monosyllables"] = sentence.apply(count_monosyllables)
    my_df["ge3syllables"] = sentence.apply(count_polysyllables, args=(3,))
    my_df["long_words"] = sentence.apply(count_long_words, args=(6,))
    my_df["infrequent100"] = sentence.apply(count_infrequent_words, args=(100,))
    my_df["infrequent1000"] = sentence.apply(count_infrequent_words, args=(1000,))

    matrix = my_df.to_numpy()
    if normalize:
        matrix = scale_linear_bycolumn(matrix)
    return matrix


def scale_linear_bycolumn(rawpoints, high=100.0, low=0.0):
    # Source: https://gist.github.com/perrygeo/4512375
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)


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


def count_pronouns(sentence):
    # TODO write desc
    number_of_pronouns = 0
    number_of_pronouns += len(
        re.findall(
            r"(?<!\w)ich(?!\w)"
            + r"|(?<!\w)du(?!\w)"
            + r"|(?<!\w)er(?!\w)"
            + r"|(?<!\w)sie(?!\w)"
            + r"|(?<!\w)es(?!\w)"
            + r"|(?<!\w)[mds]einer(?!\w)"
            + r"|(?<!\w)[mdw]ir(?!\w)"
            + r"|(?<!\w)ih[nm](?!\w)"
            + r"|(?<!\w)ihr\w{,2}(?!\w)"
            + r"|(?<!\w)uns(?!\w)"
            + r"|(?<!\w)euch(?!\w)"
            + r"|(?<!\w)ihnen(?!\w)"
            + r"|(?<!\w)[mds]ich(?!\w)"
            + r"|(?<!\w)wessen(?!\w)"
            + r"|(?<!\w)we[rnm](?!\w)"
            + r"|(?<!\w)was(?!\w)"
            + r"|(?<!\w)welche[rnms]?(?!\w)"
            + r"|(?<!\w)etwas(?!\w)"
            + r"|(?<!\w)nichts(?!\w)"
            + r"|(?<!\w)jemand\w{,2}(?!\w)"
            + r"|(?<!\w)jede\w?(?!\w)"
            + r"|(?<!\w)irgend\w{,6}(?!\w)"
            + r"|(?<!\w)man(?!\w)"
            + r"|(?<!\w)[mds]ein\w{,2}(?!\w)"
            + r"|(?<!\w)unser\w{,2}(?!\w)"
            + r"|(?<!\w)euer(?!\w)"
            + r"|(?<!\w)eur\w{1,2}(?!\w)"
            + r"|, de[nmr](?!\w)"
            + r"|, die(?!\w)"
            + r"|, das(?!\w)"
            + r"|, denen(?!\w)",
            sentence,
        )
    )
    return number_of_pronouns


def count_definite_articles(sentence):
    # TODO write desc
    number_of_definite_articles = 0
    number_of_definite_articles += len(
        re.findall(
            r"(?<!\w)de[rsnm](?!\w)|(?<!\w)das(?!\w)|(?<!\w)die(?!\w)",
            sentence,
        )
    )
    return number_of_definite_articles


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
