# Automatic Complexity Assessment of German Sentences
## Team Members
Leo Nguyen </br>
Raoul Berger </br>
Konrad Straube </br>
Till Nocher </br>

## Mail Addresses
Leo.Nguyen@gmx.de </br>
raoulb97@gmail.com </br>
konrad.straube@outlook.com </br>
nocher@cl.uni-heidelberg.de </br>


## Pretrained models:
pretrained BERT from [Deepset AI](https://deepset.ai/german-bert) </br>
pretrained word2vec from [NLPL repository](http://vectors.nlpl.eu/repository/) (model ID: 45)

## Additional Corpora Used
* TextComplexityDE19
  7 levels of difficulty
  
  1100 Wikipedia articles, 100 of them in Simple German

* Deutsche Welle - Deutsch Lernen
  2 levels of difficulty

* WEEBIT
  English news corpus
  
  5 levels of difficulty, 625 documents each

## Utilized libraries

antlr4-python3-runtime 4.8,
appdirs 1.4.4,
beautifulsoup4 4.9.3,
black 20.8b1,
blis 0.7.4,
bs4 0.0.1,
bz2file 0.98,
cached-property 1.5.2,
catalogue 2.0.1,
certifi 2020.12.5,
cffi 1.14.5,
cfgv 3.2.0,
chardet 4.0.0,
click 7.1.2,
cycler 0.10.0,
cymem 2.0.5,
Cython 0.29.21,
dataclasses 0.6,
distlib 0.3.1,
fairseq 0.10.2,
fastBPE 0.1.0,
filelock 3.0.12,
gensim 3.8.3,
gitdb 4.0.5,
GitPython 3.1.13,
google-trans-new 1.1.9,
h5py 3.1.0,
hydra-core 1.0.6,
identify 1.5.13,
idna 2.10,
importlib-metadata 3.4.0,
importlib-resources 5.1.0,
Jinja2 2.11.3,
joblib 1.0.1,
kiwisolver 1.3.1,
langdetect 1.0.8,
lxml 4.6.2,
MarkupSafe 1.1.1,
matplotlib 3.3.4,
murmurhash 1.0.5,
mypy-extensions 0.4.3,
nlpaug 1.1.2,
nltk 3.5,
nodeenv 1.5.0,
numexpr 2.7.2,
numpy 1.20.1,
omegaconf 2.0.6,
packaging 20.9,
pandas 1.2.2,
pathspec 0.8.1,
pathy 0.4.0,
Pillow 8.1.0,
plac 1.1.3,
pluggy 0.13.1,
portalocker 2.2.1,
pre-commit 2.10.1,
preshed 3.0.5,
py 1.10.0,
pycparser 2.20,
pydantic 1.7.3,
pygit 0.1,
pyparsing 2.4.7,
Pyphen 0.10.0,
python-dateutil 2.8.1,
pytz 2021.1,
PyYAML 5.4.1,
regex 2020.11.13,
requests 2.25.1,
sacrebleu 1.5.0,
sacremoses 0.0.43,
scikit-learn 0.24.1,
scipy 1.6.1,
six 1.15.0,
sklearn 0.0,
smart-open 3.0.0,
smmap 3.0.5,
soupsieve 2.2,
spacy 3.0.3,
spacy-legacy 3.0.1,
srsly 2.4.0,
stop-words 2018.7.23,
tables 3.6.1,
textstat 0.7.0,
thinc 8.0.1,
threadpoolctl 2.1.0,
tokenizers 0.10.1,
toml 0.10.2,
torch 1.7.1,
tox 3.22.0,
tqdm 4.57.0,
transformers 4.3.2,
translate 3.5.0,
typed-ast 1.4.2,
typer 0.3.2,
typing-extensions 3.7.4.3,
urllib3 1.26.3,
virtualenv 20.4.2,
wasabi 0.8.2,
zipp 3.4.0


## Setup

### Install dependencies
Install all necessary dependencies with:

> pipenv install 

### Download datasets: 

> pipenv run main --download all

To download a specific dataset, replace 'all' with ['TextComplexityDE19', 'Weebit', 'dw']

### Preprocessing and Augmentation
Run preprocessing and augmentation on datasets and save results in h5 file:

> pipenv run main --create_h5 --filename example.h5 

Additional tags: 
- --dset with argument 0 = 'TextComplexityDE19', 1 = 'Weebit', 2 = 'dw'. Example: --dset 012 for all datasets.
- --lemmatization
- --stemming
- --random_swap
- --random_deletion

Example: apply lemmatization

> pipenv run main --create_h5 --filename example.h5 --lemmatization

Note: basic preprocessing will always be applied 


## Usage

Run experiment for specific vectorizer and regression method:

> pipenv run main --experiment evaluate --filename example.h5 --vectorizer option --method option 

Addtional tag: --engineered_features (concatenate engineered features to sentence vector)

Options:

- vectorizer: 'tfidf', 'count', 'hash', 'word2vec', 'pretrained_word2vec'
- method: 'linear', 'lasso', 'ridge', 'elastic-net', 'random-forest' 

Run all combination of vectorizers and regression methods with and without engineered features:

> pipenv run main --experiment compare_all --filename example.h5

Run pretrained BERT + 3-layer regression network:

> pipenv run main --experiment train_net --filename example.h5

Additional tag: 
- --save_name name (name to save trained model under, used for training multiple models without overwriting the previous one. Default: name specified with --filename
- --engineered_features (concatenate engineered features to sentence vector)

If multiple datasets were used, you have to specify conditional training by providing the tag --multiple_datasets.

The tag --pretask [pretask_epoch, pretask_file] will overwrite the --multiple_datasets tag. In that case, instead of conditional training, the model will be first trained on a pretask (on the provided pretask_file for the given pretask_epoch) and than fine-tuned on the dataset provided by --filename. Note that the first layer of the model will be freezed after the pretask. To allow fine-tuning the first layer, use the tag --no_freeze. 

Hyperparameter tuning for word2vec: linear search along hyperparameter (generate plots and results saved to txt file)

> pipenv run main --search [hyperparameter, start, end, step, model, filename]

- hyperparameter: 'feature', 'window', 'count', 'epochs', 'lr' or 'min_lr' </br>
- start: start value of linear search </br>
- end: end value of linear search </br>
- step: step size of linear search </br>
- model: only option so far 'word2vec' </br>
- filename: h5 filename to load data from </br>

Note: experiment results are saved in folders 'result', 'figures' and 'models'

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)







