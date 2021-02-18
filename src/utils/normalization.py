import re


def normalize_sentence(
    sentence, keep_numbers=False, hyphens_are_separators=True, keep_commas=False
):
    """Normalizes sentences, meaning it decapitalizes letters, removes whitespace
    sequences, removes punctuation and removes numbers. Then returns the normalized
    sentence.

    Keyword arguments:
    sentence -- a series of sentences, for example as in a single column from a
                              pandas dataframe, to be normalized
    keep_numbers -- (optional) set true to keep numbers in the sentence instead of
                              removing them (default False)
    hyphens_are_separators -- (optional) if true, a hyphenated word is counted as 2
                              words (e.g. e-sports -> e sports), otherwise as one
                              (e-sports -> esports) (default True)
    keep_commas -- (optional, default F) don't remove commas with other punct.
    """
    normalized_sentence = sentence.str.lower()
    if not keep_numbers:
        normalized_sentence = normalized_sentence.apply(remove_numbers)
    normalized_sentence = normalized_sentence.apply(
        remove_punctuation, args=(hyphens_are_separators, keep_commas)
    )
    normalized_sentence = normalized_sentence.apply(remove_whitespace)
    return normalized_sentence


def remove_numbers(string):
    """removes numbers from a string (simple regex replacing d with nothing) and
    returns the string

    Keyword arguments:
    string -- the string to remove numbers from
    """
    return re.sub(r"\d", "", string)


def remove_punctuation(string, hyphens_are_separators=True, keep_commas=False):
    """removes punctuation from a string (simple regex replacing everything but w and
    s with nothing) and returns the string

    Keyword arguments:
    string -- the string to remove punctuation from
    hyphens_are_separators -- (optional) replace hyphens with a space first (creates a
    space in hyphenated words instead of concatenating them) (default True)
    keep_commas - (optional) don't delete commas if true (default False)
    """
    if hyphens_are_separators:
        string = re.sub(r"\-", " ", string)
    if not keep_commas:
        return re.sub(r"[^\w\s]", "", string)
    else:
        return re.sub(r"[^\w\s,]", "", string)


def remove_whitespace(string):
    """if the string contains whitespace sequences, all whitespace is replaced by a
    single space.

    Keyword arguments:
    string -- the string to remove unnecessary whitespace from
    """
    return re.sub(r"\s+", " ", string)
