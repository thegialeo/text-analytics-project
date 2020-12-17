# Exploring TextComplexityDE
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# written by Konrad, 16-Dec-20 - 17-Dec-20

import re
from os import path

import matplotlib.pyplot as plt
import pandas as pd
# %%
import scipy.stats

# %%
# load TextComplexityDE dataset
df_all = pd.read_excel(
    path.join("data", "TextComplexityDE19.xlsx"),
    sheet_name=2, header=1)
df_all.columns = df_all.columns.str.lower()


# %%
# create columns normalized_text and character_count
def remove_numbers(string):
    return re.sub(r'\d', '', string)


def remove_punctuation(string, hyphens_are_separators=True):
    if hyphens_are_separators:
        string = re.sub(r'\-', ' ', string)
    return re.sub(r'[^\w\s]', '', string)


def remove_whitespace(string):
    return re.sub(r'\s+', ' ', string)


def find_longest(list):
    length = 0
    for i in range(len(list)):
        if len(list[i]) > length:
            length = len(list[i])
    return length


def count_syllables(list):
    cc_pattern = re.compile("[^aeiouyäöü]{2,}")
    sentence_syllables = 0
    for word in list:
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


def count_polysyllables(list):
    cc_pattern = re.compile("[^aeiouyäöü]{2,}")
    polysyllables = 0
    for word in list:
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
        if word_syllables > 1:
            polysyllables += 1
    return polysyllables


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
    normalized_sentence = normalized_sentence.apply(
        remove_punctuation, args=(hyphens_are_separators,))
    normalized_sentence = normalized_sentence.apply(remove_whitespace)
    return normalized_sentence


#df_all['normalized_sentence'] = df_all['sentence'].str.lower()
# the assumption here is that numbers aren't read and understood in the foreign language anyway, so they don't contribute to complexity or lack of understandability
#df_all['normalized_sentence'] = df_all['normalized_sentence'].apply(remove_numbers)
#df_all['normalized_sentence'] = df_all['normalized_sentence'].apply(remove_punctuation)
#df_all['normalized_sentence'] = df_all['normalized_sentence'].apply(remove_whitespace)

df_all['normalized_sentence'] = normalize_sentence(df_all['sentence'])

df_all['word_count'] = df_all['normalized_sentence'].str.split().str.len()
df_all['syllable_count'] = df_all['normalized_sentence'].str.split().apply(count_syllables)
df_all['letter_count'] = df_all['normalized_sentence'].str.count(r'\w')
df_all['polysyllables_count'] = df_all['normalized_sentence'].str.split().apply(
    count_polysyllables)
df_all['flesch_score'] = 206.835 - 1.015 * df_all['word_count'] - \
    84.6 * (df_all['syllable_count'] * 1.0 / df_all['word_count'])
df_all['ari'] = 4.71 * (df_all['letter_count']
                        * 1.0 / df_all['word_count']) + .5 * df_all['word_count'] - 21.43
df_all['mean_word_length'] = (df_all['letter_count'] * 1.0) / df_all['word_count']
df_all['mean_squared_word_length'] = (
    (df_all['letter_count'] * 1.0) ** 2) / df_all['word_count']
df_all['max_word_length'] = df_all['normalized_sentence'].str.split().apply(find_longest)


# %%
string = 'beim aufblasen entsteht eine kugelform die wasserversorgung erfolgte über brunnen etwa jahre ist es her seit die sumerer das'
string2 = 'wasser laptop grüsse flasche kaiser packen'
testwort = 'donaudampfschifffahrtsgesellschaftskapitän'

for word in testwort.split():
    print(word, count_syllables([word]))


# %%
print(df_all['word_count'].sum())


# %%
df_all['rmos_r'] = df_all['mos_r'].round().astype('int64')
df_all['rmos_u'] = df_all['mos_u'].round().astype('int64')
df_all['rmos_l'] = df_all['mos_l'].round().astype('int64')


# %%
# R Readability/Complexity, U Understandability, L Lexical difficulty
df_all.head()


# %%
df_all.sort_values(by="ari", ascending=False).head()


# %%
df_wikipedia = df_all[df_all['article_id'] < 24]
df_leichte = df_all[df_all['article_id'] > 23]

print('Sentences sourced from Wikipedia:', df_wikipedia['id'].size)
print('Sentences sourced from Leichte Sprache:', df_leichte['id'].size)


# %%
score_counts_r_all = df_all['rmos_r'].value_counts().sort_index()
print("sentences rated by score:")
print(score_counts_r_all)
plot = score_counts_r_all.plot.pie(subplots=True, figsize=(5, 5))


# %%
print('wikipedia:')
score_counts_r_wikipedia = df_wikipedia['rmos_r'].value_counts().sort_index()
print(score_counts_r_wikipedia)
#plot = score_counts_r_wikipedia.plot.pie(subplots=True, figsize=(5, 5))

print('\nleichte sprache:')
score_counts_r_leichte = df_leichte['rmos_r'].value_counts().sort_index()
print(score_counts_r_leichte)
#plot = score_counts_r_leichte.plot.pie(subplots=True, figsize=(5, 5))

# %% [markdown]
# Among the 900 sentences sourced from Wikipedia, there are few that scored a 1 or a 6 in complexity and none(!) that scored a 7.
# The 100 sentences sourced from Leichte Sprache are dominated by 1 scores.

# %%
print('median complexity score for all sentences:', df_all['mos_r'].median())
print('median complexity score for wikipedia sentences:',
      df_wikipedia['mos_r'].median())
print('median complexity score for leichte sentences:', df_leichte['mos_r'].median())
print()

print('median understandability score for all sentences:', df_all['mos_u'].median())
print(
    'median understandability score for wikipedia sentences:',
    df_wikipedia['mos_u'].median())
print(
    'median understandability score for leichte sentences:',
    df_leichte['mos_u'].median())
print()

print('median lexical difficulty score for all sentences:', df_all['mos_l'].median())
print(
    'median lexical difficulty score for wikipedia sentences:',
    df_wikipedia['mos_l'].median())
print(
    'median lexical difficulty score for leichte sentences:',
    df_leichte['mos_l'].median())


# %%


# %%
fig = plt.figure(figsize=(10, 3))

plt.subplot(131)
plot_colors = ["#457cd6", "#e34262"]
plt.title(r'complexity scores')
plt.hist([df_wikipedia['mos_r'], df_leichte['mos_r']], 35,
         stacked=True, density=True, color=plot_colors)
plt.xticks(range(1, 8))
plt.xlim([0.5, 7.5])
plt.ylim([0.0, 0.82])
plt.yticks([0.0, 0.82], ['', ''])
#meanx = df_all['mos_r'].mean()
medianx = df_all['mos_r'].median()
# plt.axvline(meanx, color='#6d8c32', linestyle='dashdot', alpha=.8)
plt.axvline(medianx, color='#2c1b2e', linestyle='--', alpha=.8)

plt.subplot(132)
plot_colors = ["#457cd6", "#e34262"]
plt.title(r'understandability scores')
plt.hist([df_wikipedia['mos_u'], df_leichte['mos_u']], 35,
         stacked=True, density=True, color=plot_colors)
plt.xticks(range(1, 8))
plt.xlim([0.5, 7.5])
plt.ylim([0.0, 0.82])
plt.yticks([0.0, 0.82], ['', ''])
medianx = df_all['mos_u'].median()
plt.axvline(medianx, color='#2c1b2e', linestyle='--', alpha=.8)
plt.text(medianx + .1, .63, 'median', rotation=90, color='#2c1b2e', alpha=.8)

plt.subplot(133)
plot_colors = ["#457cd6", "#e34262"]
plt.title(r'lexical difficulty scores')
plt.hist([df_wikipedia['mos_l'], df_leichte['mos_l']], 35, stacked=True,
         density=True, color=plot_colors, label=["Wikipedia", "Leichte Sprache"])
plt.legend(loc="upper right", title="Source")
plt.xticks(range(1, 8))
plt.xlim([0.5, 7.5])
plt.ylim([0.0, 0.82])
plt.yticks([0.0, 0.82], ['', ''])
medianx = df_all['mos_l'].median()
plt.axvline(medianx, color='#2c1b2e', linestyle='--', alpha=.8)

plt.tight_layout()


# %%


# %%
x_col = 'ari'  # try word_count	letter_count	mean_word_length	mean_squared_word_length	max_word_length   syllable_count   ari
y_col = 'mos_r'


# %%
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


# %%

# %%
