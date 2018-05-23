import itertools
from functools import reduce

import numpy as np
import torch
import json

import torch.nn as nn
import torch.nn.init
from data_packer import CRFDataset_WC


zip = getattr(itertools, 'izip', zip)


def to_scalar(var):
    """change the first element of a tensor to scalar
    """
    return var.view(-1).data.tolist()[0]


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum

    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M


def encode2char_safe(input_lines, char_dict):
    """
    get char representation of lines

    args:
        input_lines (list of strings) : input corpus
        char_dict (dictionary) : char-level dictionary

    """
    unk = char_dict['<u>']
    forw_lines = [list(map(lambda m: list(map(lambda t: char_dict.get(t, unk), m)), line)) for line in input_lines]
    return forw_lines


def concatChar(input_lines, char_dict):
    """
    concat char into string

    args:
        input_lines (list of list of char) : input corpus
        char_dict (dictionary) : char-level dictionary

    """
    features = [[char_dict[' ']] + list(reduce(lambda x, y: x + [char_dict[' ']] + y, sentence)) + [char_dict['\n']] for sentence in input_lines]
    return features


def get_crf_scrf_label():
    SCRF_l_map = {}
    SCRF_l_map['PER'] = 0
    SCRF_l_map['LOC'] = 1
    SCRF_l_map['ORG'] = 2
    SCRF_l_map['MISC'] = 3
    CRF_l_map = {}
    for pre in ['S-', 'B-', 'I-', 'E-']:
        for suf in SCRF_l_map.keys():
            CRF_l_map[pre + suf] = len(CRF_l_map)
    SCRF_l_map['<START>'] = 4
    SCRF_l_map['<STOP>'] = 5
    SCRF_l_map['O'] = 6
    CRF_l_map['<start>'] = len(CRF_l_map)
    CRF_l_map['<pad>'] = len(CRF_l_map)
    CRF_l_map['O'] = len(CRF_l_map)

    return CRF_l_map, SCRF_l_map


def encode_safe(input_lines, word_dict, unk):
    """
    encode list of strings into word-level representation with unk

    """

    lines = list(map(lambda t: list(map(lambda m: word_dict.get(m, unk), t)), input_lines))
    return lines


def encode(input_lines, word_dict):
    """
    encode list of strings into word-level representation

    """

    lines = list(map(lambda t: list(map(lambda m: word_dict[m], t)), input_lines))
    return lines

def encode_SCRF(input_lines, word_dict):
    """
    encode list of strings into word-level representation

    """

    lines = list(map(lambda t: list(map(lambda m: [m[0], m[1], word_dict[m[2]], word_dict[m[3]]], t)), input_lines))
    return lines


def generate_corpus_char(lines, if_shrink_c_feature=False, c_thresholds=1, if_shrink_w_feature=False, w_thresholds=1):
    """
    generate label, feature, word dictionary, char dictionary and label dictionary

    args:
        lines : corpus
        if_shrink_c_feature: whether shrink char-dictionary
        c_threshold: threshold for shrinking char-dictionary
        if_shrink_w_feature: whether shrink word-dictionary
        w_threshold: threshold for shrinking word-dictionary
        
    """
    features, labels, feature_map, label_map = generate_corpus(lines, if_shrink_feature=if_shrink_w_feature, thresholds=w_thresholds)
    char_count = dict()
    for feature in features:
        for word in feature:
            for tup in word:
                if tup not in char_count:
                    char_count[tup] = 0
                else:
                    char_count[tup] += 1
    if if_shrink_c_feature:
        shrink_char_count = [k for (k, v) in iter(char_count.items()) if v >= c_thresholds]
        char_map = {shrink_char_count[ind]: ind for ind in range(0, len(shrink_char_count))}
    else:
        char_map = {k: v for (v, k) in enumerate(char_count.keys())}

    # add three special chars
    char_map['<u>'] = len(char_map)
    char_map[' '] = len(char_map)
    char_map['\n'] = len(char_map)
    return features, labels, feature_map, label_map, char_map


def shrink_features(feature_map, features, thresholds):
    """
    filter un-common features by threshold

    """

    feature_count = {k: 0 for (k, v) in iter(feature_map.items())}
    for feature_list in features:
        for feature in feature_list:
            feature_count[feature] += 1
    shrinked_feature_count = [k for (k, v) in iter(feature_count.items()) if v >= thresholds]
    feature_map = {shrinked_feature_count[ind]: (ind + 1) for ind in range(0, len(shrinked_feature_count))}

    feature_map['<unk>'] = 0
    feature_map['<eof>'] = len(feature_map)
    return feature_map


def generate_corpus(lines, if_shrink_feature=False, thresholds=1):
    """
    generate label, feature, word dictionary and label dictionary

    args:
        lines : corpus
        if_shrink_feature: whether shrink word-dictionary
        threshold: threshold for shrinking word-dictionary
        
    """
    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    feature_map = dict()
    label_map = dict()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            if line[0] not in feature_map:
                feature_map[line[0]] = len(feature_map) + 1 #0 for unk
            tmp_ll.append(line[-1])
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(iob_iobes(tmp_ll))
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(iob_iobes(tmp_ll))
    for ls in labels:
        for l in ls:
            if l not in label_map:
                label_map[l] = len(label_map)
    label_map['<start>'] = len(label_map)
    label_map['<pad>'] = len(label_map)
    if if_shrink_feature:
        feature_map = shrink_features(feature_map, features, thresholds)
    else:
        feature_map['<unk>'] = 0
        feature_map['<eof>'] = len(feature_map)

    return features, labels, feature_map, label_map


def read_corpus(lines):
    """
    convert corpus into features and labels

    """

    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            tmp_ll.append(line[-1])
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(iob_iobes(tmp_ll))
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(iob_iobes(tmp_ll))

    return features, labels


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    iob2(tags)
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:
            tags[i] = 'B' + tag[1:]
    return True


def load_embedding(emb_file, delimiter, feature_map, full_feature_set, unk, emb_len, shrink_to_train=False, shrink_to_corpus=False):
    """
    load embedding, indoc words would be listed before outdoc words

    args: 
        emb_file: path to embedding file
        delimiter: delimiter of lines
        feature_map: word dictionary
        full_feature_set: all words in the corpus
        caseless: convert into casesless style
        unk: string for unknown token
        emb_len: dimension of embedding vectors
        shrink_to_train: whether to shrink out-of-training set or not
        shrink_to_corpus: whether to shrink out-of-corpus or not

    """

    feature_set = set([key.lower() for key in feature_map])
    full_feature_set = set([key.lower() for key in full_feature_set])

    
    word_dict = {v:(k+1) for (k,v) in enumerate(feature_set - set(['<unk>']))}
    word_dict['<unk>'] = 0

    in_doc_freq_num = len(word_dict)
    rand_embedding_tensor = torch.FloatTensor(in_doc_freq_num, emb_len)
    init_embedding(rand_embedding_tensor)

    indoc_embedding_array = list()
    indoc_word_array = list()
    outdoc_embedding_array = list()
    outdoc_word_array = list()

    for line in open(emb_file, 'r'):
        line = line.split(delimiter)
        vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        if shrink_to_train and line[0] not in feature_set:
            continue

        if line[0] == unk:
            rand_embedding_tensor[0] = torch.FloatTensor(vector) #unk is 0
        elif line[0] in word_dict:
            rand_embedding_tensor[word_dict[line[0]]] = torch.FloatTensor(vector)
        elif line[0] in full_feature_set:
            indoc_embedding_array.append(vector)
            indoc_word_array.append(line[0])
        elif not shrink_to_corpus:
            outdoc_word_array.append(line[0])
            outdoc_embedding_array.append(vector)
    
    embedding_tensor_0 = torch.FloatTensor(np.asarray(indoc_embedding_array))

    if not shrink_to_corpus:
        embedding_tensor_1 = torch.FloatTensor(np.asarray(outdoc_embedding_array))
        word_emb_len = embedding_tensor_0.size(1)
        assert(word_emb_len == emb_len)

    if shrink_to_corpus:
        embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0], 0)
    else:
        embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0, embedding_tensor_1], 0)

    for word in indoc_word_array:
        word_dict[word] = len(word_dict)
    in_doc_num = len(word_dict)
    if  not shrink_to_corpus:
        for word in outdoc_word_array:
            word_dict[word] = len(word_dict)

    return word_dict, embedding_tensor, in_doc_num


def calc_threshold_mean(features):
    """
    calculate the threshold for bucket by mean
    """
    lines_len = list(map(lambda t: len(t) + 1, features))
    average = int(sum(lines_len) / len(lines_len))
    lower_line = list(filter(lambda t: t < average, lines_len))
    upper_line = list(filter(lambda t: t >= average, lines_len))
    lower_average = int(sum(lower_line) / len(lower_line))
    upper_average = int(sum(upper_line) / len(upper_line))
    max_len = max(lines_len)
    return [lower_average, average, upper_average, max_len]


def CRFtag_to_SCRFtag(inputs):
    alltags = []
    for input in inputs:
        tags = []
        beg = 0
        oldtag = '<START>'
        for i, tag in enumerate(input):
            if tag == u'O':
                tags.append((i, i, oldtag, tag))
                oldtag = tag
            if tag[0] == u'S':
                tags.append((i, i, oldtag, tag[2:]))
                oldtag = tag[2:]
            if tag[0] == u'B':
                beg = i
            if tag[0] == u'E':
                tags.append((beg, i, oldtag, tag[2:]))
                oldtag = tag[2:]
        alltags.append(tags)
    return alltags


def construct_bucket_mean_vb_wc(word_features, input_label, label_dict, SCRF_label_dict, char_dict, word_dict, SCRF_stop_tag, ALLOW_SPANLEN=6, train_set=False):
    """
    Construct bucket by mean for viterbi decode, word-level and char-level
    """

    SCRFtags = CRFtag_to_SCRFtag(input_label)
    labels = encode(input_label, label_dict)
    SCRFlabels = encode_SCRF(SCRFtags, SCRF_label_dict)

    new_SCRFlabels = []
    new_labels = []
    new_word_features = []
    nolycrf_labels = []
    nolycrf_word_features = []
    nolycrf_SCRFlabels = []
    if train_set:
        for word, SCRFlabel, label in zip(word_features, SCRFlabels, labels):
            keep = True
            for t in SCRFlabel:
                if t[1]-t[0] >= ALLOW_SPANLEN:
                    keep = False
                    break
            if keep:
                new_word_features.append(word)
                new_labels.append(label)
                new_SCRFlabels.append(SCRFlabel)
            else:
                nolycrf_labels.append(label)
                nolycrf_word_features.append(word)
                nolycrf_SCRFlabels.append(SCRFlabel)
    else:
        new_word_features = word_features
        new_labels = labels
        new_SCRFlabels = SCRFlabels

    char_features = encode2char_safe(new_word_features, char_dict)
    fea_len = [list(map(lambda t: len(t) + 1, f)) for f in char_features]
    forw_features = concatChar(char_features, char_dict)
    new_labels = list(map(lambda t: [label_dict['<start>']] + list(t), new_labels))
    thresholds = calc_threshold_mean(fea_len)
    new_word_features = list(map(lambda t: list(map(lambda x: x.lower(), t)), new_word_features))
    new_word_features = encode_safe(new_word_features, word_dict, word_dict['<unk>'])
    dataset = construct_bucket_vb_wc(new_word_features, forw_features, fea_len, new_labels, new_SCRFlabels, char_features,
                                  thresholds, word_dict['<eof>'], char_dict['\n'],
                                  label_dict['<pad>'], len(label_dict), SCRF_stop_tag)

    if train_set:
        if nolycrf_word_features:
            nolycrf_char_features = encode2char_safe(nolycrf_word_features, char_dict)
            nolycrf_fea_len = [list(map(lambda t: len(t) + 1, f)) for f in nolycrf_char_features]
            nolycrf_forw_features = concatChar(nolycrf_char_features, char_dict)
            nolycrf_labels = list(map(lambda t: [label_dict['<start>']] + list(t), nolycrf_labels))
            nolycrf_thresholds = [max(list(map(lambda t: len(t) + 1, nolycrf_fea_len)))]
            nolycrf_word_features = list(map(lambda t: list(map(lambda x: x.lower(), t)), nolycrf_word_features))
            nolycrf_word_features = encode_safe(nolycrf_word_features, word_dict, word_dict['<unk>'])
            nolycrf_dataset = construct_bucket_vb_wc(nolycrf_word_features, nolycrf_forw_features, nolycrf_fea_len,
                                                     nolycrf_labels, nolycrf_SCRFlabels, nolycrf_char_features,
                                                     nolycrf_thresholds, word_dict['<eof>'], char_dict['\n'],
                                                     label_dict['<pad>'], len(label_dict), SCRF_stop_tag)
            return dataset, nolycrf_dataset
        else:
            return dataset, None
    else:
        return dataset


def accumulate(iterator):
    total = 0
    for item in iterator:
        total += item
        yield total


def construct_bucket_vb_wc(word_features, forw_features, fea_len, input_labels, SCRFlabels, char_features, thresholds, pad_word_feature, pad_char_feature, pad_label, label_size, SCRF_stop_tag):
    """
    Construct bucket by thresholds for viterbi decode, word-level and char-level
    """
    word_max_len = max([len(c) for c_fs in char_features for c in c_fs])
    buckets = [[[], [], [], [], [], [], [], [], [], [], []] for ind in range(len(thresholds))]
    buckets_len = [0 for ind in range(len(thresholds))]
    for f_f, f_l in zip(forw_features, fea_len):
        cur_len_1 = len(f_l) + 1
        idx = 0
        while thresholds[idx] < cur_len_1:
            idx += 1
        tmp_concat_len = len(f_f) + thresholds[idx] - len(f_l)
        if buckets_len[idx] < tmp_concat_len:
            buckets_len[idx] = tmp_concat_len
    for f_f, f_l, w_f, i_l, s_l, c_f in zip(forw_features, fea_len, word_features, input_labels, SCRFlabels, char_features):
        cur_len = len(f_l)
        idx = 0
        cur_len_1 = cur_len + 1
        cur_scrf_len = len(s_l)
        cur_scrf_len_1 = cur_scrf_len + 1
        w_l = max(f_l)-1

        while thresholds[idx] < cur_len_1:
            idx += 1

        padded_feature = f_f + [pad_char_feature] * (buckets_len[idx] - len(f_f))  # pad feature with <'\n'>, at least one

        padded_feature_len = f_l + [1] * (thresholds[idx] - len(f_l)) # pad feature length with <'\n'>, at least one

        padded_feature_len_cum = list(accumulate(padded_feature_len)) # start from 0, but the first is ' ', so the position need not to be -1
        buckets[idx][0].append(padded_feature) # char
        buckets[idx][1].append(padded_feature_len_cum)
        buckets[idx][2].append(padded_feature[::-1])
        buckets[idx][3].append([buckets_len[idx] - 1] + [buckets_len[idx] - 1 - tup for tup in padded_feature_len_cum[:-1]])
        buckets[idx][4].append(w_f + [pad_word_feature] * (thresholds[idx] - cur_len)) #word
        buckets[idx][5].append([i_l[ind] * label_size + i_l[ind + 1] for ind in range(0, cur_len)] + [i_l[cur_len] * label_size + pad_label] + [pad_label * label_size + pad_label] * (thresholds[idx] - cur_len_1))  # has additional start, label
        buckets[idx][6].append([1] * cur_len_1 + [0] * (thresholds[idx] - cur_len_1))  # has additional start, mask
        buckets[idx][7].append([len(f_f) + thresholds[idx] - len(f_l), cur_len_1, cur_scrf_len_1, w_l])
        buckets[idx][8].append(s_l + [[s_l[-1][1]+1, s_l[-1][1]+1, s_l[-1][-1], SCRF_stop_tag]] + [[0, 0, 0, 0] for _ in range(thresholds[idx]-cur_scrf_len_1)])
        buckets[idx][9].append([1] * cur_scrf_len_1 + [0] * (thresholds[idx] - cur_scrf_len_1))
        buckets[idx][10].append([c + [pad_char_feature]*(word_max_len-len(c)) for c in c_f] + [[pad_char_feature]*word_max_len]*(thresholds[idx]-cur_len))
    bucket_dataset = [CRFDataset_WC(torch.LongTensor(bucket[0]), torch.LongTensor(bucket[1]),
                                    torch.LongTensor(bucket[2]), torch.LongTensor(bucket[3]),
                                    torch.LongTensor(bucket[4]), torch.LongTensor(bucket[5]),
                                    torch.ByteTensor(bucket[6]), torch.LongTensor(bucket[7]),
                                    torch.LongTensor(bucket[8]), torch.ByteTensor(bucket[9]),
                                    torch.LongTensor(bucket[10]))
                                    for bucket in buckets]
    return bucket_dataset


def tuple_to_seq_BIOES(tuples, id_to_tag):

    sentlen = max([tuple[1] for tuple in tuples]) + 1
    seq = [None for _ in range(sentlen)]
    for tuple in tuples:
        if id_to_tag[tuple[-1]] == 'O':
                for i in range(tuple[0], tuple[1]+1):
                    seq[i] = 'O'
        else:
            if tuple[1]-tuple[0] == 0:
                seq[tuple[0]] = 'S-' + id_to_tag[tuple[-1]]
            elif tuple[1]-tuple[0] >= 1:
                seq[tuple[0]] = 'B-' + id_to_tag[tuple[-1]]
                seq[tuple[1]] = 'E-' + id_to_tag[tuple[-1]]
                for i in range(tuple[0] + 1, tuple[1]):
                    seq[i] = 'I-' + id_to_tag[tuple[-1]]
    return seq

def find_length_from_labels(labels, label_to_ix):
    """
    find length of unpadded features based on labels
    """
    end_position = len(labels) - 1
    for position, label in enumerate(labels):
        if label == label_to_ix['<pad>']:
            end_position = position
            break
    return end_position


def revlut(lut):
    return {v: k for k, v in lut.items()}

def iobes_to_spans(sequence, lut, strict_iob2=False):
    """
    convert to iobes to span
    """
    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = lut[y]
        if label.startswith('B-'):
            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i]

        elif label.startswith('S-'):
            if current is not None:
                chunks.append('@'.join(current))
                current = None
            base = label.replace('S-', '')
            chunks.append('@'.join([base, '%d' % i]))

        elif label.startswith('I-'):
            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning')
                    current = [base, '%d' % i]
            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning')

        elif label.startswith('E-'):
            if current is not None:
                base = label.replace('E-', '')
                if base == current[0]:
                    current.append('%d' % i)
                    chunks.append('@'.join(current))
                    current = None
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning')
                    current = [base, '%d' % i]
                    chunks.append('@'.join(current))
                    current = None
            else:
                current = [label.replace('E-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning')
                chunks.append('@'.join(current))
                current = None
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)


def save_checkpoint(state, track_list, filename):
    """
    save checkpoint
    """
    with open(filename+'.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename+'.model')

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
    # if input_lstm.bidirectional:
    #     for ind in range(0, input_lstm.num_layers):
    #         weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
    #         bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    #         nn.init.uniform(weight, -bias, bias)
    #         weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
    #         bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    #         nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        # if input_lstm.bidirectional:
        #     for ind in range(0, input_lstm.num_layers):
        #         weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
        #         weight.data.zero_()
        #         weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        #         weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
        #         weight.data.zero_()
        #         weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def crf_to_scrf(decoded_crf, r_l_map, scrf_l_map):
    """
    crf labels to scrf labels

    """

    input_label = []
    for seq in decoded_crf:
        sentencecrf = []
        for i, l in enumerate(seq):
            tag = r_l_map[l]
            if tag == '<pad>':
                break
            sentencecrf.append(tag)
        input_label.append(sentencecrf)
    SCRFtags = CRFtag_to_SCRFtag(input_label)
    SCRFlabels = encode_SCRF(SCRFtags, scrf_l_map)
    maxl_1 = max([j[1] for i in SCRFlabels for j in i]) + 2
    scrfdata = []
    masks = []
    for s_l in SCRFlabels:
        cur_scrf_len = len(s_l)
        s_l_pad = s_l + \
                  [[0, 0, 0, 0] for _ in range(maxl_1 - cur_scrf_len)]
        mask = [1] * cur_scrf_len + [0] * (maxl_1 - cur_scrf_len)
        scrfdata.append(s_l_pad)
        masks.append(mask)
    scrfdata = torch.cuda.LongTensor(scrfdata)
    masks = torch.cuda.ByteTensor(masks)
    return scrfdata, masks

def scrf_to_crf(decoded_scrf, l_map):
    """
    scrf labels to crf labels

    """
    label_size = len(l_map)
    crf_labels = []
    pad_label = l_map['<pad>']
    for i_l in decoded_scrf:
        sent_labels = [l_map['<start>']]
        for label in i_l:
            if label != l_map['<pad>']:
                sent_labels.append(label)
            else:
                break
        crf_labels.append(sent_labels)

    crfdata = []
    masks = []
    maxl_1 = max([len(i) for i in crf_labels])
    for i_l in crf_labels:
        cur_len_1 = len(i_l)
        cur_len = cur_len_1 - 1
        i_l_pad = [i_l[ind] * label_size + i_l[ind + 1] for ind in range(0, cur_len)] + [i_l[cur_len] * label_size + pad_label] + [
                    pad_label * label_size + pad_label] * (maxl_1 - cur_len_1)

        mask = [1] * cur_len_1 + [0] * (maxl_1 - cur_len_1)
        crfdata.append(i_l_pad)
        masks.append(mask)
    crfdata = torch.cuda.LongTensor(crfdata).transpose(0,1).unsqueeze(2)
    masks = torch.cuda.ByteTensor(masks).transpose(0,1)
    return crfdata, masks

def decode_with_crf(crf, word_reps, mask_v, l_map):
    """
    decode with viterbi algorithm and return score

    """

    seq_len = word_reps.size(0)
    bat_size = word_reps.size(1)
    decoded_crf = crf.decode(word_reps, mask_v)
    scores = crf.cal_score(word_reps).data
    mask_v = mask_v.data
    decoded_crf = decoded_crf.data
    decoded_crf_withpad = torch.cat((torch.cuda.LongTensor(1,bat_size).fill_(l_map['<start>']), decoded_crf), 0)
    decoded_crf_withpad = decoded_crf_withpad.transpose(0,1).cpu().numpy()
    label_size = len(l_map)

    bi_crf = []
    cur_len = decoded_crf_withpad.shape[1]-1
    for i_l in decoded_crf_withpad:
        bi_crf.append([i_l[ind] * label_size + i_l[ind + 1] for ind in range(0, cur_len)] + [
            i_l[cur_len] * label_size + l_map['<pad>']])
    bi_crf = torch.cuda.LongTensor(bi_crf).transpose(0,1).unsqueeze(2)

    tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, bi_crf).view(seq_len, bat_size)  # seq_len * bat_size
    tg_energy = tg_energy.transpose(0,1).masked_select(mask_v.transpose(0,1))
    tg_energy = tg_energy.cpu().numpy()
    masks = mask_v.sum(0)
    crf_result_scored_by_crf = []
    start = 0
    for i, mask in enumerate(masks):
        end = start + mask
        crf_result_scored_by_crf.append(tg_energy[start:end].sum())
        start = end
    crf_result_scored_by_crf = np.array(crf_result_scored_by_crf)
    return decoded_crf.cpu().transpose(0,1).numpy(), crf_result_scored_by_crf

def rescored_with_scrf(decoded_crf, r_l_map, SCRF_l_map, decoder_scrf):
    """
    re-score crf deocded labels with scrf

    """
    scrfdata, masks = crf_to_scrf(decoded_crf, r_l_map, SCRF_l_map)
    scrf_batch_score = decoder_scrf.get_logloss_numerator(scrfdata, decoder_scrf.SCRF_scores.data, masks)
    masks = masks.sum(1)
    scrf_batch_score = scrf_batch_score.cpu().numpy()
    crf_result_scored_by_scrf = []
    start = 0
    for i, mask in enumerate(masks):
        end = start + mask
        crf_result_scored_by_scrf.append(scrf_batch_score[start:end].sum())
        start = end
    crf_result_scored_by_scrf = np.array(crf_result_scored_by_scrf)
    return crf_result_scored_by_scrf

def rescored_with_crf(decoded_scrf, l_map, scores):
    """
    re-score scrf decoded labels with crf

    """
    scrfdata, masks = scrf_to_crf(decoded_scrf, l_map)
    seq_len = scores.size(0)
    bat_size = scores.size(1)
    tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, scrfdata).view(seq_len, bat_size)  # seq_len * bat_size
    crf_batch_score = tg_energy.transpose(0,1).masked_select(masks.transpose(0,1))
    masks = masks.sum(0)
    scrf_result_scored_by_crf = []
    start = 0
    for i, mask in enumerate(masks):
        end = start + mask
        scrf_result_scored_by_crf.append(crf_batch_score[start:end].sum())
        start = end

    scrf_result_scored_by_crf = torch.cat(scrf_result_scored_by_crf).cpu().data.numpy()
    return scrf_result_scored_by_crf

