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
To download a specific dataset, replace 'all' with ['TextComplexityDE19', 'Weebit', 'dw']: 

> pipenv run main --download all

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


## Usage





## Project State








