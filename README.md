# Hybrid semi-Markov CRF

HSCRF achieves F1 score of 91.38+/-0.10 on the CoNLL 2003 NER dataset, without using any additional corpus or resource.

## Installation

### PyTorch

The code is based on PyTorch. You can find installation instructions [here](http://pytorch.org/).

### Dependencies

The code is written in Python 2.7 and pytorch 0.2.0. Its dependencies are summarized in the file ```requirements.txt```. You can install these dependencies like this:

```
pip install -r requirements.txt
```

### Code reference

[LM-LSTM-CRF](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF)

## Usage

```
CUDA_VISIBLE_DEVICES=0 python train.py --char_lstm --high_way
```

### word embeddings

Glove: You can find the pre-trained word embedding [here](https://nlp.stanford.edu/projects/glove/),
and place glove.6B.100d.txt in `data/`.

## Reference

```
TO BE ADDED
```