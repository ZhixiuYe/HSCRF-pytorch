from __future__ import print_function
import torch
import codecs
import argparse
import json

import model.utils as utils
from model.evaluator import evaluator
from model.model import ner_model
from model.data_packer import Repack

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating LM-BLSTM-CRF')
    parser.add_argument('--load_arg', default='./checkpoint/6365035.json', help='path to arg json')
    parser.add_argument('--load_check_point', default='./checkpoint/6365035.model',
                        help='path to model checkpoint file')
    parser.add_argument('--dev_file', default='data/eng.testa',
                        help='path to development file, if set to none, would use dev_file path in the checkpoint file')
    parser.add_argument('--test_file', default='data/eng.testb',
                        help='path to test file, if set to none, would use test_file path in the checkpoint file')
    args = parser.parse_args()


    with open(args.load_arg, 'r') as f:
        jd = json.load(f)
    jd = jd['args']

    checkpoint_file = torch.load(args.load_check_point)
    f_map = checkpoint_file['f_map']
    CRF_l_map = checkpoint_file['CRF_l_map']
    c_map = checkpoint_file['c_map']
    in_doc_words = checkpoint_file['in_doc_words']
    SCRF_l_map = checkpoint_file['SCRF_l_map']
    ALLOW_SPANLEN = checkpoint_file['ALLOW_SPANLEN']

    with codecs.open(args.dev_file, 'r', 'utf-8') as f:
        dev_lines = f.readlines()

    with codecs.open(args.test_file, 'r', 'utf-8') as f:
        test_lines = f.readlines()


    dev_features, dev_labels = utils.read_corpus(dev_lines)
    test_features, test_labels = utils.read_corpus(test_lines)

    dev_dataset = utils.construct_bucket_mean_vb_wc(dev_features, dev_labels, CRF_l_map, SCRF_l_map, c_map, f_map, SCRF_stop_tag=SCRF_l_map['<STOP>'], train_set=False)
    test_dataset = utils.construct_bucket_mean_vb_wc(test_features, test_labels, CRF_l_map, SCRF_l_map, c_map, f_map, SCRF_stop_tag=SCRF_l_map['<STOP>'], train_set=False)

    dev_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in dev_dataset]
    test_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in test_dataset]

    print('build model')
    model = ner_model(jd['word_embedding_dim'], jd['word_hidden_dim'], jd['word_lstm_layers'],
                      len(f_map), len(c_map), jd['char_embedding_dim'], jd['char_lstm_hidden_dim'],
                      jd['cnn_filter_num'], jd['char_lstm_layers'], jd['char_lstm'],jd['dropout_ratio'],
                      jd['high_way'], jd['highway_layers'], CRF_l_map['<start>'], CRF_l_map['<pad>'],
                      len(CRF_l_map), SCRF_l_map, jd['scrf_dense_dim'], in_doc_words,
                      jd['index_embeds_dim'], jd['allowspan'], SCRF_l_map['<START>'], SCRF_l_map['<STOP>'],
                      jd['grconv'])

    print('load model')
    model.load_state_dict(checkpoint_file['state_dict'])

    model.cuda()
    packer = Repack()

    evaluator = evaluator(packer, CRF_l_map, SCRF_l_map)


    print('dev...')
    dev_f1_crf, dev_pre_crf, dev_rec_crf, dev_acc_crf, dev_f1_scrf, dev_pre_scrf, dev_rec_scrf, dev_acc_scrf, dev_f1_jnt, dev_pre_jnt, dev_rec_jnt, dev_acc_jnt = \
            evaluator.calc_score(model, dev_dataset_loader)
    print('test...')
    test_f1_crf, test_pre_crf, test_rec_crf, test_acc_crf, test_f1_scrf, test_pre_scrf, test_rec_scrf, test_acc_scrf, test_f1_jnt, test_pre_jnt, test_rec_jnt, test_acc_jnt = \
            evaluator.calc_score(model, test_dataset_loader)

    print(' dev_f1: %.4f\n' % (dev_f1_crf))
    print(' dev_f1_scrf: %.4f\n' % (dev_f1_scrf))
    print(' dev_f1_jnt: %.4f\n' % (dev_f1_jnt))

    print(' test_f1: %.4f\n' % (test_f1_crf))
    print(' test_f1_scrf: %.4f\n' % (test_f1_scrf))
    print(' test_f1_jnt: %.4f\n' % (test_f1_jnt))


