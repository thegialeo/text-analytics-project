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


## Existing Code Fragments
## Utilized libraries

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
- dset with argument 0 = 'TextComplexityDE19', 1 = 'Weebit', 2 = 'dw'. Example: --dset 012 for all datasets.
- lemmatization
- stemming
- random_swap
- random_deletion

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

Note: experiment results are saved in folders 'result', 'figures' and 'models'

Run pretrained BERT + 3-layer regression network:

> pipenv run main --experiment train_net --filename example.h5

Additional tag: --save_name name (name to save trained model under, used for training multiple models without overwriting the previous one. Default: name specified with --filename









