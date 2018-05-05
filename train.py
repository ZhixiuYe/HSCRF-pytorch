from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
import codecs
import model.utils as utils
from model.evaluator import evaluator
from model.model import ner_model
from model.data_packer import Repack

import argparse
import os
import sys
from tqdm import tqdm
import itertools
import functools
import numpy as np


# seed = int(np.random.uniform(0,1)*10000000)
seed = 5703958
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
print('seed: ', seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with LM-LSTM-CRF together with Language Model')
    parser.add_argument('--emb_file', default='./data/glove.6B.100d.txt', help='path to pre-trained embedding')
    parser.add_argument('--train_file', default='./data/eng.train', help='path to training file')
    parser.add_argument('--dev_file', default='./data/eng.testa', help='path to development file')
    parser.add_argument('--test_file', default='./data/eng.testb', help='path to test file')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--unk', default='unk', help='unknow-token in pre-trained embedding')
    parser.add_argument('--char_lstm_hidden_dim', type=int, default=300, help='dimension of char-level lstm layer for language model')
    parser.add_argument('--word_hidden_dim', type=int, default=300, help='dimension of word-level lstm layer')
    parser.add_argument('--dropout_ratio', type=float, default=0.55, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=150, help='maximum epoch number')
    parser.add_argument('--least_epoch', type=int, default=75, help='minimum epoch number')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop epoch number')
    parser.add_argument('--start_epoch', type=int, default=0, help='start point of epoch')
    parser.add_argument('--checkpoint', default='./checkpoint/', help='checkpoint path')
    parser.add_argument('--word_embedding_dim', type=int, default=100, help='dimension of word embedding')
    parser.add_argument('--char_embedding_dim', type=int, default=30, help='dimension of character embedding')
    parser.add_argument('--scrf_dense_dim', type=int, default=100, help='dimension of scrf features')
    parser.add_argument('--index_embeds_dim', type=int, default=10, help='dimension of index embedding')
    parser.add_argument('--cnn_filter_num', type=int, default=30, help='the number of cnn filters')
    parser.add_argument('--char_lstm_layers', type=int, default=1, help='number of char level layers for language model')
    parser.add_argument('--word_lstm_layers', type=int, default=1, help='number of word level layers')
    parser.add_argument('--lr', type=float, default=0.015, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')
    parser.add_argument('--load_check_point', default='', help='path previous checkpoint that want to be loaded')
    parser.add_argument('--load_opt', action='store_true', help='also load optimizer from the checkpoint')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='clip grad at')
    parser.add_argument('--mini_count', type=float, default=5, help='thresholds to replace rare words with <unk>')
    parser.add_argument('--high_way', action='store_true', help='use highway layers')
    parser.add_argument('--highway_layers', type=int, default=1, help='number of highway layers')
    parser.add_argument('--shrink_embedding', action='store_true', help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus)')
    parser.add_argument('--model_name', default='HSCRF', help='model name')
    parser.add_argument('--char_lstm', action='store_true', help='use lstm for characters embedding or not')
    parser.add_argument('--allowspan', type=int, default=6, help='allowed max segment length')
    parser.add_argument('--grconv', action='store_true', help='use grconv')

    args = parser.parse_args()

    CRF_l_map, SCRF_l_map = utils.get_crf_scrf_label()

    print('setting:')
    print(args)

    print('loading corpus')
    with codecs.open(args.train_file, 'r', 'utf-8') as f:
        lines = f.readlines()
    with codecs.open(args.dev_file, 'r', 'utf-8') as f:
        dev_lines = f.readlines()
    with codecs.open(args.test_file, 'r', 'utf-8') as f:
        test_lines = f.readlines()

    dev_features, dev_labels = utils.read_corpus(dev_lines)
    test_features, test_labels = utils.read_corpus(test_lines)

    if args.load_check_point:
        if os.path.isfile(args.load_check_point):
            print("loading checkpoint: '{}'".format(args.load_check_point))
            checkpoint_file = torch.load(args.load_check_point)
            args.start_epoch = checkpoint_file['epoch']
            f_map = checkpoint_file['f_map']
            c_map = checkpoint_file['c_map']
            in_doc_words = checkpoint_file['in_doc_words']
            train_features, train_labels = utils.read_corpus(lines)
        else:
            print("no checkpoint found at: '{}'".format(args.load_check_point))
            sys.exit()
    else:
        print('constructing coding table')

        train_features, train_labels, f_map, _, c_map = \
            utils.generate_corpus_char(lines, if_shrink_c_feature=True,
                                       c_thresholds=args.mini_count,
                                       if_shrink_w_feature=False)

        f_set = {v for v in f_map}

        f_map = utils.shrink_features(f_map, train_features, args.mini_count)
        dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_features), f_set)
        dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_features), dt_f_set)

        f_map, embedding_tensor, in_doc_words = utils.load_embedding(args.emb_file, ' ', f_map, dt_f_set, args.unk, args.word_embedding_dim, shrink_to_corpus=args.shrink_embedding)

        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_labels))
        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_labels), l_set)

    print('constructing dataset')
    dataset, dataset_onlycrf = utils.construct_bucket_mean_vb_wc(train_features, train_labels, CRF_l_map, SCRF_l_map, c_map, f_map, SCRF_stop_tag=SCRF_l_map['<STOP>'], ALLOW_SPANLEN=args.allowspan, train_set=True)
    dev_dataset = utils.construct_bucket_mean_vb_wc(dev_features, dev_labels, CRF_l_map, SCRF_l_map, c_map, f_map, SCRF_stop_tag=SCRF_l_map['<STOP>'], train_set=False)
    test_dataset = utils.construct_bucket_mean_vb_wc(test_features, test_labels, CRF_l_map, SCRF_l_map, c_map, f_map, SCRF_stop_tag=SCRF_l_map['<STOP>'], train_set=False)

    dataset_loader = [torch.utils.data.DataLoader(tup, args.batch_size, shuffle=True, drop_last=False) for tup in dataset]
    dataset_loader_crf = [torch.utils.data.DataLoader(tup, 3, shuffle=True, drop_last=False) for tup in dataset_onlycrf]
    dev_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in dev_dataset]
    test_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in test_dataset]

    print('building model')
    model = ner_model(args.word_embedding_dim, args.word_hidden_dim, args.word_lstm_layers, len(f_map),
                      len(c_map), args.char_embedding_dim, args.char_lstm_hidden_dim, args.cnn_filter_num,
                      args.char_lstm_layers, args.char_lstm, args.dropout_ratio, args.high_way, args.highway_layers,
                      CRF_l_map['<start>'], CRF_l_map['<pad>'], len(CRF_l_map), SCRF_l_map, args.scrf_dense_dim,
                      in_doc_words,args.index_embeds_dim, args.allowspan, SCRF_l_map['<START>'], SCRF_l_map['<STOP>'], args.grconv)

    if args.load_check_point:
        model.load_state_dict(checkpoint_file['state_dict'])
    else:
        model.word_rep.load_pretrained_word_embedding(embedding_tensor)
        model.word_rep.rand_init()

    optimizer = optim.SGD(model.parameters(),
                           lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters())

    if args.load_check_point and args.load_opt:
        optimizer.load_state_dict(checkpoint_file['optimizer'])

    model.cuda()
    packer = Repack()

    tot_length = sum(map(lambda t: len(t), dataset_loader))

    best_dev_f1_jnt = float('-inf')
    best_test_f1_crf = float('-inf')
    best_test_f1_scrf = float('-inf')
    best_test_f1_jnt = float('-inf')
    start_time = time.time()
    early_stop_epochs = 0
    epoch_list = range(args.start_epoch, args.start_epoch + args.epoch)

    evaluator = evaluator(packer, CRF_l_map, SCRF_l_map)

    for epoch_idx, args.start_epoch in enumerate(epoch_list):

        epoch_loss = 0
        model.train()

        for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, SCRF_labels, mask_SCRF_labels, cnn_features in tqdm(
                itertools.chain.from_iterable(dataset_loader_crf), mininterval=2,
                desc=' - Tot it %d (epoch %d)' % (tot_length, args.start_epoch), leave=False, file=sys.stderr):

                f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, SCRF_labels, mask_SCRF_labels, cnn_features = packer.repack(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, SCRF_labels, mask_SCRF_labels, cnn_features, test=False)

                optimizer.zero_grad()

                loss = model(f_f, f_p, b_f, b_p, w_f, cnn_features, tg_v, mask_v,
                      mask_v.long().sum(0), SCRF_labels, mask_SCRF_labels, onlycrf=True)

                epoch_loss += utils.to_scalar(loss)

                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
                optimizer.step()

        for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, SCRF_labels, mask_SCRF_labels, cnn_features in tqdm(
                itertools.chain.from_iterable(dataset_loader), mininterval=2,
                desc=' - Tot it %d (epoch %d)' % (tot_length, args.start_epoch), leave=False, file=sys.stderr):

            f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, SCRF_labels, mask_SCRF_labels, cnn_features = packer.repack(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v, SCRF_labels, mask_SCRF_labels, cnn_features, test=False)
            optimizer.zero_grad()

            loss = model(f_f, f_p, b_f, b_p, w_f, cnn_features, tg_v, mask_v,
                         mask_v.long().sum(0), SCRF_labels, mask_SCRF_labels, onlycrf=False)

            epoch_loss += utils.to_scalar(loss)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            optimizer.step()

        epoch_loss /= tot_length
        print('epoch_loss: ', epoch_loss)

        utils.adjust_learning_rate(optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))


        dev_f1_crf, dev_pre_crf, dev_rec_crf, dev_acc_crf, dev_f1_scrf, dev_pre_scrf, dev_rec_scrf, dev_acc_scrf, dev_f1_jnt, dev_pre_jnt, dev_rec_jnt, dev_acc_jnt = \
                evaluator.calc_score(model, dev_dataset_loader)

        if dev_f1_jnt > best_dev_f1_jnt:
            early_stop_epochs = 0
            test_f1_crf, test_pre_crf, test_rec_crf, test_acc_crf, test_f1_scrf, test_pre_scrf, test_rec_scrf, test_acc_scrf, test_f1_jnt, test_pre_jnt, test_rec_jnt, test_acc_jnt = \
                        evaluator.calc_score(model, test_dataset_loader)

            best_test_f1_crf = test_f1_crf
            best_test_f1_scrf = test_f1_scrf

            best_dev_f1_jnt = dev_f1_jnt
            best_test_f1_jnt = test_f1_jnt

            try:
                utils.save_checkpoint({
                        'epoch': args.start_epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'f_map': f_map,
                        'c_map': c_map,
                        'SCRF_l_map': SCRF_l_map,
                        'CRF_l_map': CRF_l_map,
                        'in_doc_words': in_doc_words,
                        'ALLOW_SPANLEN': args.allowspan
                    }, {'args': vars(args)
                        }, args.checkpoint + str(seed))
            except Exception as inst:
                    print(inst)

        else:
            early_stop_epochs += 1

        print('best_test_f1_crf is: %.4f' % (best_test_f1_crf))
        print('best_test_f1_scrf is: %.4f' % (best_test_f1_scrf))
        print('best_test_f1_jnt is: %.4f' % (best_test_f1_jnt))

        print('epoch: ' + str(args.start_epoch) + '\t in ' + str(args.epoch) + ' take: ' + str(
            time.time() - start_time) + ' s')

        sys.stdout.flush()

        if early_stop_epochs >= args.early_stop and epoch_idx > args.least_epoch:
            break


    print('setting:')
    print(args)
    print('seed: ', seed)