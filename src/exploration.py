# Exploring TextComplexityDE
# written by Konrad, 16-Dec-20 - 17-Dec-20

import re
from os import path

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats

def remove_numbers(string):
    return re.sub(r'\d', '', string)

def remove_punctuation(string, hyphens_are_separators=True):
    if hyphens_are_separators:
        string = re.sub(r'\-', ' ', string)
    return re.sub(r'[^\w\s]', '', string)

def remove_whitespace(string):
    return re.sub(r'\s+', ' ', string)

def count_syllables(words):
    """Given a list of words, counts the syllables in each and returns the sum. Counting syllables doesn't necessarily give very good results

    Keyword arguments:
    words -- list of words (use split() on sentences, maybe)
    """
    cc_pattern = re.compile("[^aeiouyäöü]{2,}") ## two consonants in a row
    sentence_syllables = 0
    for word in words:
        word_syllables = 1
        current_pos = len(word) - 1
        while current_pos >= 0:
            current_character = word[current_pos]
            current_pos -= 1
            if current_character in 'aeiouyäöü':
                if current_pos <= 0:
                    break
                else:
                    current_character = word[current_pos]
                    if current_character not in 'aeiouyäöü':
                        word_syllables += 1
                    current_pos -= 1
        if cc_pattern.match(word) and len(word) > 2:
            word_syllables -= 1
        sentence_syllables += word_syllables
    return sentence_syllables

def count_polysyllables(sentence, threshold=2):
    """Given a sentence, computes and returns the number of polysyllabic words containing at least a certain, optionally specified amount of syllables

    Keyword arguments:
    sentence -- sentence or series of sentences
    threshold -- (optional) this function will count words with at least this many syllables (default 2)
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
            if current_character in 'aeiouyäöü':
                if current_pos <= 0:
                    break
                else:
                    current_character = word[current_pos]
                    if current_character not in 'aeiouyäöü':
                        word_syllables += 1
                    current_pos -= 1
        if cc_pattern.match(word) and len(word) > 2:
            word_syllables -= 1
        if word_syllables >= threshold:
            polysyllables += 1
    return polysyllables

def count_monosyllables(sentence):
    """Given a sentence, computes and returns the number of monosyllabic words (words which are just 1 syllable long) contained within

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
            if current_character in 'aeiouyäöü':
                if current_pos <= 0:
                    break
                else:
                    current_character = word[current_pos]
                    if current_character not in 'aeiouyäöü':
                        word_syllables += 1
                    current_pos -= 1
        if cc_pattern.match(word) and len(word) > 2:
            word_syllables -= 1
        if word_syllables == 1:
            monosyllables += 1
    return monosyllables

def count_long_words(sentence, length):
    """Given a sentence, computes and returns the number of words equal to or longer than the provided threshold length

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

def normalize_sentence(sentence, keep_numbers=False, hyphens_are_separators=True):
    """Normalizes sentences, meaning it decapitalizes letters, removes whitespace sequences, removes punctuation and removes numbers. Then returns the normalized sentence.

    Keyword arguments:
    sentence -- a series of sentences, for example as in a single column from a pandas dataframe, to be normalized
    keep_numbers -- (optional) set true to keep numbers in the sentence instead of removing them (default False)
    hyphens_are_separators -- (optional) if true, a hyphenated word is counted as 2 words (e.g. e-sports -> e sports), otherwise as one (e-sports -> esports) (default True)
    """
    normalized_sentence = sentence.str.lower()
    if not keep_numbers:
        normalized_sentence = normalized_sentence.apply(remove_numbers)
    normalized_sentence = normalized_sentence.apply(remove_punctuation, args=(hyphens_are_separators,))
    normalized_sentence = normalized_sentence.apply(remove_whitespace)
    return normalized_sentence

def flesch_reading_ease(word_count, syllable_count, deutsch=True):
    """Given a number of words and number of syllables in a sentence, this will compute the flesch readin ease score for the sentence

    Keyword arguments:
    word_count -- number of words in the sentence
    syllable_count -- number of syllables in the sentence
    deutsch -- (optional) use german instead of english formula (default True)
    """
    if deutsch:
        return 180 - 1.0 * word_count - 58.5 * syllable_count / word_count
    else:
        return 206.835 - 1.015 * word_count - 84.6 * syllable_count / word_count

if __name__ == "__main__":
    # load TextComplexityDE dataset
    df_all = pd.read_excel(path.join("src", "data", "TextComplexityDE19.xlsx"), engine='openpyxl', sheet_name=2, header=1)

    df_all.columns = df_all.columns.str.lower()

    df_all['normalized_sentence'] = normalize_sentence(df_all['sentence'])

    df_all['word_count'] = df_all['normalized_sentence'].str.split().str.len()
    df_all['letter_count'] = df_all['normalized_sentence'].str.count(r'\w')
    df_all['syllable_count'] = df_all['normalized_sentence'].str.split().apply(count_syllables)
    df_all['monosyllables_count'] = df_all['normalized_sentence'].apply(count_monosyllables)
    df_all['two_syllables_count'] = df_all['normalized_sentence'].apply(count_polysyllables)  #counts two OR MORE
    df_all['three_syllables_count'] = df_all['normalized_sentence'].apply(count_polysyllables, args=(3,))
    df_all['long_words_count'] = df_all['normalized_sentence'].apply(count_long_words, args=(6,))

    df_all['fre'] = flesch_reading_ease(df_all['word_count'], df_all['syllable_count'], deutsch=False)
    df_all['fre_deutsch'] = flesch_reading_ease(df_all['word_count'], df_all['syllable_count'], deutsch=True)

    df_all['fkgl'] = .39 * df_all['word_count'] + 11.8 * df_all['syllable_count'] / df_all['word_count'] - 15.59 #flesch kincaid grade level
    df_all['ari'] = 4.71 * df_all['letter_count'] / df_all['word_count'] + .5 * df_all['word_count'] - 21.43
    df_all['mean_word_length'] = (df_all['letter_count'] * 1.0) / df_all['word_count']

    string = 'beim aufblasen entsteht eine kugelform die wasserversorgung erfolgte über brunnen etwa jahre ist es her seit die sumerer das'
    for word in string.split():
        print(word, count_syllables([word]))

    print('words in dataset:', df_all['word_count'].sum())

    # R Readability/Complexity, U Understandability, L Lexical difficulty
    df_all.head()

    df_wikipedia = df_all[df_all['article_id'] < 24]
    df_leichte = df_all[df_all['article_id'] > 23]

    print('Sentences sourced from Wikipedia:', df_wikipedia['id'].size)
    print('Sentences sourced from Leichte Sprache:', df_leichte['id'].size)

    score_distribution = df_all['mos_r'].round().astype('int64').value_counts().sort_index()
    print('score distribution:\n', score_distribution)
    plot = score_distribution.plot.pie(subplots=True, figsize=(5, 5))

    print('median complexity score for all sentences:', df_all['mos_r'].median())
    print('median complexity score for wikipedia sentences:',df_wikipedia['mos_r'].median())
    print('median complexity score for leichte sentences:', df_leichte['mos_r'].median())

    fig = plt.figure(figsize=(10, 3))

    plt.subplot(131)
    plot_colors = ["#457cd6", "#e34262"]
    plt.title(r'complexity scores')
    plt.hist([df_wikipedia['mos_r'], df_leichte['mos_r']], 35, stacked=True, density=True, color=plot_colors)
    plt.xticks(range(1, 8))
    plt.xlim([0.5, 7.5])
    plt.ylim([0.0, 0.82])
    plt.yticks([0.0, 0.82], ['', ''])
    medianx = df_all['mos_r'].median()
    plt.axvline(medianx, color='#2c1b2e', linestyle='--', alpha=.8)

    plt.subplot(132)
    plt.title(r'understandability scores')
    plt.hist([df_wikipedia['mos_u'], df_leichte['mos_u']], 35, stacked=True, density=True, color=plot_colors)
    plt.xticks(range(1, 8))
    plt.xlim([0.5, 7.5])
    plt.ylim([0.0, 0.82])
    plt.yticks([0.0, 0.82], ['', ''])
    medianx = df_all['mos_u'].median()
    plt.axvline(medianx, color='#2c1b2e', linestyle='--', alpha=.8)
    plt.text(medianx + .1, .63, 'median', rotation=90, color='#2c1b2e', alpha=.8)

    plt.subplot(133)
    plt.title(r'lexical difficulty scores')
    plt.hist([df_wikipedia['mos_l'], df_leichte['mos_l']], 35, stacked=True, density=True, color=plot_colors, label=["Wikipedia", "Leichte Sprache"])
    plt.legend(loc="upper right", title="Source")
    plt.xticks(range(1, 8))
    plt.xlim([0.5, 7.5])
    plt.ylim([0.0, 0.82])
    plt.yticks([0.0, 0.82], ['', ''])
    medianx = df_all['mos_l'].median()
    plt.axvline(medianx, color='#2c1b2e', linestyle='--', alpha=.8)

    plt.tight_layout()

    feature_list = ['word_count', 'letter_count', 'fre', 'fre_deutsch', 'fkgl', 'ari']
    x_col = 'fre_deutsch'  
    y_col = 'mos_r'

    x = df_all[x_col]
    y = df_all[y_col]

    slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=0, marker='x', alpha=.43, label='Data points')
    ax.plot(x, intercept + slope * x, label=line)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(facecolor='white')
    plt.show()
