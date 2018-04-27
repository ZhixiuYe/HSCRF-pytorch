# Hybrid semi-Markov CRF

HSCRF achieves F1 score of 91.38+/-0.10 on the CoNLL 2003 NER dataset, without using any additional corpus or resource.

## Installation

### PyTorch

The code is based on PyTorch. You can find installation instructions [here](http://pytorch.org/).

### Dependencies

The code is written in Python 2.7. Its dependencies are summarized in the file ```requirements.txt```. You can install these dependencies like this:

```
pip install -r requirements.txt
```

## Usage

```
CUDA_VISIBLE_DEVICES=0 python train.py --char_lstm
```

### word embeddings

Glove: https://nlp.stanford.edu/projects/glove/

## Reference

```
TO BE ADDED
```
