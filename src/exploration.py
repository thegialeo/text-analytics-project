# Exploring TextComplexityDE

import re
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

import word_rarity


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
        wordlist = word_rarity.uni_leipzig_top100de()
    elif size == 1000:
        wordlist = word_rarity.uni_leipzig_top1000de()
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
    """
    normalized_sentence = sentence.str.lower()
    if not keep_numbers:
        normalized_sentence = normalized_sentence.apply(remove_numbers)
    normalized_sentence = normalized_sentence.apply(
        remove_punctuation, args=(hyphens_are_separators, keep_commas)
    )
    normalized_sentence = normalized_sentence.apply(remove_whitespace)
    return normalized_sentence


def flesch_reading_ease(word_count, syllable_count, deutsch=True):
    """Given a number of words and number of syllables in a sentence, this will
    compute the flesch reading ease score for the sentence

    Keyword arguments:
    word_count -- number of words in the sentence
    syllable_count -- number of syllables in the sentence
    deutsch -- (optional) use german instead of english formula (default True)
    """
    if deutsch:
        return 180 - 1.0 * word_count - 58.5 * syllable_count / word_count
    else:
        return 206.835 - 1.015 * word_count - 84.6 * syllable_count / word_count


def flesch_kincaid_grade_level(word_count, syllable_count):
    """Given a number of words and number of syllables in a sentence, computes the
    flesch kincaid grade level.

    Keyword arguments:
    word_count -- number of words in the sentence
    syllable_count -- number of syllables in the sentence
    """
    return 0.39 * word_count + 11.8 * syllable_count / word_count - 15.59


def automated_readability_index(letter_count, word_count):
    """Given a number of letters and number of words in a sentence, computes the
    automated readability index

    Keyword arguments:
    letter_count -- number of letters in the sentence
    word_count -- number of words in the sentence
    """
    return 4.71 * letter_count / word_count + 0.5 * word_count - 21.43


def gunning_fox_index(word_count, polysyllables_count):
    """Given a number of words and the number of words with at least 3 syllables in a
    sentence, compute the gunning-fox-index.
    Note that this was originally intended for the english language, and for a section
    of text containing at least 100 words.

    Keyword arguments:
    word_count -- number of words in the sentence
    polysyllables_count -- number of words with at least 3 syllables in the sentence`
    """
    return (word_count + polysyllables_count) * 0.4


def smog(polysyllables_count):
    """Given the number of words with at least 3 syllables in a text, compute the SMOG
    grade.
    Note that this was originally intended for the english language, and for a section
    of text containing at least 30 sentences.

    Keyword arguments:
    polysyllables_count -- number of words with at least 3 syllables in the sentence`
    """
    return 1.043 * (30.0 * polysyllables_count) ** 0.5 + 3.1291


def coleman_liau_index(letter_count, word_count):
    """Given a number of letters and number of words in a sentence, computes the
    coleman-liau index

    Keyword arguments:
    letter_count -- number of letters in the sentence
    word_count -- number of words in the sentence
    """

    return (
        0.0588 * letter_count / (word_count * 100) - 0.296 / (word_count * 100) - 15.8
    )


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


def wiener_sachtextformel2(polysyllables_count, word_count, long_words_count):
    """Computes the second wiener sachtextformel, using the number of words with 3 or
    more syllables, the number of words and the number of words with 6 or more
    letters.

    Keyword arguments:
    polysyllables_count -- number of words with three or more syllables
    word_count -- number of words
    long_words_count -- number of words with 6 or more letters
    """

    return (
        0.2007 * polysyllables_count / word_count
        + 0.1682 * word_count
        + 0.1373 * long_words_count / word_count
        - 2.779
    )


def gulpease_index(character_count, word_count):
    return 89 - 10.0 * character_count / word_count + 300.0 / word_count


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


if __name__ == "__main__":
    # load TextComplexityDE dataset
    df_all = pd.read_excel(
        "data/TextComplexityDE19/TextComplexityDE19.xlsx",
        engine="openpyxl",
        sheet_name=2,
        header=1,
    )
    df_all.columns = df_all.columns.str.lower()

    df_all["normalized_sentence"] = normalize_sentence(df_all["sentence"])
    # print(type(df_all["normalized_sentence"]))
    df_all["normalized_sentence_with_commas"] = normalize_sentence(
        df_all["sentence"], keep_commas=True
    )

    df_all[["word_count", "letter_count"]] = count_words_and_letters(
        df_all["normalized_sentence"]
    )

    df_all["pronouns_count"] = df_all["normalized_sentence_with_commas"].apply(
        count_pronouns
    )
    df_all["def_arti_count"] = df_all["normalized_sentence_with_commas"].apply(
        count_definite_articles
    )

    df_all["words_wo_pronouns"] = (
        df_all["word_count"] - df_all["pronouns_count"] - df_all["def_arti_count"]
    )

    df_all["syllable_count"] = (
        df_all["normalized_sentence"].str.split().apply(count_syllables)
    )
    df_all["monosyllables_count"] = df_all["normalized_sentence"].apply(
        count_monosyllables
    )
    df_all["two_syllables_count"] = df_all["normalized_sentence"].apply(
        count_polysyllables, args=(2,)
    )  # counts two OR MORE
    df_all["three_syllables_count"] = df_all["normalized_sentence"].apply(
        count_polysyllables, args=(3,)
    )
    df_all["long_words_count"] = df_all["normalized_sentence"].apply(
        count_long_words, args=(6,)
    )

    df_all["infrequent_words100"] = df_all["normalized_sentence"].apply(
        count_infrequent_words, args=(100,)
    )
    df_all["infrequent_words1000"] = df_all["normalized_sentence"].apply(
        count_infrequent_words, args=(1000,)
    )

    df_all["fre"] = flesch_reading_ease(
        df_all["word_count"], df_all["syllable_count"], deutsch=False
    )
    df_all["fre_deutsch"] = flesch_reading_ease(
        df_all["word_count"], df_all["syllable_count"], deutsch=True
    )

    df_all["fkgl"] = flesch_kincaid_grade_level(
        df_all["word_count"], df_all["syllable_count"]
    )
    df_all["ari"] = automated_readability_index(
        df_all["letter_count"], df_all["word_count"]
    )
    df_all["gfi"] = gunning_fox_index(
        df_all["word_count"], df_all["three_syllables_count"]
    )
    df_all["smog"] = smog(df_all["three_syllables_count"])
    df_all["cli"] = coleman_liau_index(df_all["letter_count"], df_all["word_count"])
    df_all["wstf"] = wiener_sachtextformel(
        df_all["three_syllables_count"],
        df_all["word_count"],
        df_all["long_words_count"],
        df_all["monosyllables_count"],
    )
    df_all["wstf2"] = wiener_sachtextformel2(
        df_all["three_syllables_count"],
        df_all["word_count"],
        df_all["long_words_count"],
    )

    df_all["gi"] = gulpease_index(df_all["letter_count"], df_all["word_count"])

    df_all["mean_word_length"] = (df_all["letter_count"] * 1.0) / df_all["word_count"]

    string = (
        "beim aufblasen entsteht eine kugelform die wasserversorgung erfolgte "
        + "über brunnen etwa jahre ist es her seit die sumerer das"
    )

    stringo = "über brunnen etwa jahre ist es her seit die sumerer das"
    print("pronouns in string:", count_pronouns(stringo))
    print("definite articles in string:", count_definite_articles(stringo))

    print("-------")
    features = construct_features(df_all["normalized_sentence"])
    print(features)
    print(features.shape)
    print("-------")
    features = construct_features(df_all["normalized_sentence"], normalize=True)
    print(features)
    print(features.shape)
    print("-------")

    # print(df_all['words_wo_pronouns'].sort_values())

    # for word in string.split():
    # print(word, count_syllables([word]))

    # print("words in dataset:", df_all["word_count"].sum())

    # R Readability/Complexity, U Understandability, L Lexical difficulty
    # df_all.head()

    df_wikipedia = df_all[df_all["article_id"] < 24]
    df_leichte = df_all[df_all["article_id"] > 23]

    if False:
        print("Sentences sourced from Wikipedia:", df_wikipedia["id"].size)
        print("Sentences sourced from Leichte Sprache:", df_leichte["id"].size)

        score_distribution = (
            df_all["mos_r"].round().astype("int64").value_counts().sort_index()
        )
        print("score distribution:\n", score_distribution)
        plot = score_distribution.plot.pie(subplots=True, figsize=(5, 5))

        print("median complexity score for all sentences:", df_all["mos_r"].median())
        print(
            "median complexity score for wikipedia sentences:",
            df_wikipedia["mos_r"].median(),
        )
        print(
            "median complexity score for leichte sentences:",
            df_leichte["mos_r"].median(),
        )

    if False:
        fig = plt.figure(figsize=(10, 3))

        plt.subplot(131)
        plot_colors = ["#457cd6", "#e34262"]
        plt.title(r"complexity scores")
        plt.hist(
            [df_wikipedia["mos_r"], df_leichte["mos_r"]],
            35,
            stacked=True,
            density=True,
            color=plot_colors,
        )
        plt.xticks(range(1, 8))
        plt.xlim([0.5, 7.5])
        plt.ylim([0.0, 0.82])
        plt.yticks([0.0, 0.82], ["", ""])
        medianx = df_all["mos_r"].median()
        plt.axvline(medianx, color="#2c1b2e", linestyle="--", alpha=0.8)

        plt.subplot(132)
        plt.title(r"understandability scores")
        plt.hist(
            [df_wikipedia["mos_u"], df_leichte["mos_u"]],
            35,
            stacked=True,
            density=True,
            color=plot_colors,
        )
        plt.xticks(range(1, 8))
        plt.xlim([0.5, 7.5])
        plt.ylim([0.0, 0.82])
        plt.yticks([0.0, 0.82], ["", ""])
        medianx = df_all["mos_u"].median()
        plt.axvline(medianx, color="#2c1b2e", linestyle="--", alpha=0.8)
        plt.text(medianx + 0.1, 0.63, "median", rotation=90, color="#2c1b2e", alpha=0.8)

        plt.subplot(133)
        plt.title(r"lexical difficulty scores")
        plt.hist(
            [df_wikipedia["mos_l"], df_leichte["mos_l"]],
            35,
            stacked=True,
            density=True,
            color=plot_colors,
            label=["Wikipedia", "Leichte Sprache"],
        )
        plt.legend(loc="upper right", title="Source")
        plt.xticks(range(1, 8))
        plt.xlim([0.5, 7.5])
        plt.ylim([0.0, 0.82])
        plt.yticks([0.0, 0.82], ["", ""])
        medianx = df_all["mos_l"].median()
        plt.axvline(medianx, color="#2c1b2e", linestyle="--", alpha=0.8)

        plt.tight_layout()

    if True:
        feature_list = [
            "word_count",
            "syllable_count",
            "letter_count",
            "infrequent_words100",
            "infrequent_words1000",
            "fre",
            "fre_deutsch",
            "fkgl",
            "ari",
            "gfi",
            "smog",
            "cli",
            "wstf",
            "wstf2",
            "gi",
            "pronouns_count",
            "def_arti_count",
            "words_wo_pronouns",
        ]

        for i in range(len(feature_list)):
            slope, intercept, r, p, stderr = scipy.stats.linregress(
                df_all[feature_list[i]], df_all["mos_r"]
            )
            print(
                "correlation of ",
                feature_list[i],
                " with complexity ratings has r: ",
                r,
                ", R-squared: ",
                r ** 2,
                sep="",
            )

    if False:
        x_col = "gi"
        y_col = "mos_r"

        x = df_all[x_col]
        y = df_all[y_col]

        slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
        line = f"Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}"

        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=0, marker="x", alpha=0.43, label="Data points")
        ax.plot(x, intercept + slope * x, label=line)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(facecolor="white")
        plt.show()

        print()
