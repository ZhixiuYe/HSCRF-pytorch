from __future__ import print_function, division
import torch.nn as nn
from word_rep_layer import WORD_REP
from crf_layer import CRF
from hscrf_layer import HSCRF


class ner_model(nn.Module):

    def __init__(self, word_embedding_dim, word_hidden_dim, word_lstm_layers, vocab_size, char_size,
                 char_embedding_dim, char_lstm_hidden_dim, cnn_filter_num, char_lstm_layers, char_lstm,
                 dropout_ratio, if_highway, highway_layers, crf_start_tag, crf_end_tag, crf_target_size,
                 scrf_tag_map, scrf_dense_dim, in_doc_words, index_embeds_dim, ALLOWED_SPANLEN,
                 scrf_start_tag, scrf_end_tag, grconv):

        super(ner_model, self).__init__()

        self.char_lstm = char_lstm
        self.word_rep = WORD_REP(char_size, char_embedding_dim, char_lstm_hidden_dim, cnn_filter_num,
                 char_lstm_layers, word_embedding_dim,
                 word_hidden_dim, word_lstm_layers, vocab_size, dropout_ratio, if_highway=if_highway,
                 in_doc_words=in_doc_words, highway_layers=highway_layers, char_lstm=char_lstm)

        self.crf = CRF(crf_start_tag, crf_end_tag, word_hidden_dim, crf_target_size)

        self.hscrf = HSCRF(scrf_tag_map, word_rep_dim=word_hidden_dim, SCRF_feature_dim=scrf_dense_dim,
                           index_embeds_dim=index_embeds_dim, ALLOWED_SPANLEN=ALLOWED_SPANLEN,
                           start_id=scrf_start_tag, stop_id=scrf_end_tag,grconv=grconv)


    def forward(self, forw_sentence, forw_position, back_sentence, back_position, word_seq,
                cnn_features, crf_target, crf_mask, scrf_mask_words, scrf_target, scrf_mask_target, onlycrf=True):
        """
        calculate loss

        :param forw_sentence   (char_seq_len, batch_size) : char-level representation of sentence
        :param forw_position   (word_seq_len, batch_size) : position of blank space in char-level representation of sentence
        :param back_sentence   (char_seq_len, batch_size) : char-level representation of sentence (inverse order)
        :param back_position   (word_seq_len, batch_size) : position of blank space in inversed char-level representation of sentence
        :param word_seq        (word_seq_len, batch_size) : word-level representation of sentence
        :param cnn_features    (word_seq_len, batch_size, word_len) : char-level representation of words
        :param crf_target      (word_seq_len, batch_size, 1): labels for CRF
        :param crf_mask        (word_seq_len, batch_size) : mask for crf_target and word_seq
        :param scrf_mask_words (batch_size) : lengths of sentences
        :param scrf_target     (batch_size, tag_len, 4) : labels for SCRF
        :param scrf_mask_target(batch_size, tag_len) : mask for scrf_target
        :param onlycrf         (True or False) : whether training data is suitable for SCRF
        :return:
        """

        word_representations = self.word_rep(forw_sentence, forw_position, back_sentence, back_position, word_seq, cnn_features)
        loss_crf = self.crf(word_representations, crf_target, crf_mask)
        loss = loss_crf
        if not onlycrf:
            loss_scrf = self.hscrf(word_representations.transpose(0,1), scrf_mask_words, scrf_target, scrf_mask_target)
            loss = loss + loss_scrf
        if self.char_lstm:
            loss_lm = self.word_rep.lm_loss(forw_sentence, forw_position, back_sentence, back_position, word_seq)
            loss = loss + loss_lm
        return loss
