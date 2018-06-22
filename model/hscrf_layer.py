from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import utils


class HSCRF(nn.Module):

    def __init__(self, tag_to_ix, word_rep_dim=300, SCRF_feature_dim=100, index_embeds_dim=10, ALLOWED_SPANLEN=6, start_id=4, stop_id=5, noBIES=False, no_index=False, no_sub=False, grconv=False):
        super(HSCRF, self).__init__()

        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {v:k for k,v in self.tag_to_ix.items()}
        self.tagset_size = len(tag_to_ix)
        self.index_embeds_dim = index_embeds_dim
        self.SCRF_feature_dim = SCRF_feature_dim
        self.ALLOWED_SPANLEN = ALLOWED_SPANLEN
        self.start_id = start_id
        self.stop_id = stop_id
        self.grconv = grconv

        self.index_embeds = nn.Embedding(self.ALLOWED_SPANLEN, self.index_embeds_dim)
        self.init_embedding(self.index_embeds.weight)

        self.dense = nn.Linear(word_rep_dim, self.SCRF_feature_dim)
        self.init_linear(self.dense)

        # 4 for SBIE, 3 for START, STOP, O and 2 for START and O
        self.CRF_tagset_size = 4*(self.tagset_size-3)+2

        self.transition = nn.Parameter(
            torch.zeros(self.tagset_size, self.tagset_size))

        span_word_embedding_dim = 2*self.SCRF_feature_dim + self.index_embeds_dim
        self.new_hidden2CRFtag = nn.Linear(span_word_embedding_dim, self.CRF_tagset_size)
        self.init_linear(self.new_hidden2CRFtag)

        if self.grconv:
            self.Wl = nn.Linear(self.SCRF_feature_dim, self.SCRF_feature_dim)
            self.Wr = nn.Linear(self.SCRF_feature_dim, self.SCRF_feature_dim)
            self.Gl = nn.Linear(self.SCRF_feature_dim, 3*self.SCRF_feature_dim)
            self.Gr = nn.Linear(self.SCRF_feature_dim, 3*self.SCRF_feature_dim)
            self.toSCRF = nn.Linear(self.SCRF_feature_dim, self.tagset_size)
            self.init_linear(self.Wl)
            self.init_linear(self.Wr)
            self.init_linear(self.Gl)
            self.init_linear(self.Gr)
            self.init_linear(self.toSCRF)


    def init_embedding(self, input_embedding):
        """
        Initialize embedding

        """
        bias = np.sqrt(3.0 / input_embedding.size(1))
        nn.init.uniform(input_embedding, -bias, bias)

    def init_linear(self, input_linear):
        """
        Initialize linear transformation

        """
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def get_logloss_denominator(self, scores, mask):
        """
        calculate all path scores of SCRF with dynamic programming

        args:
            scores (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : features for SCRF
            mask   (batch_size) : mask for words

        """

        logalpha = Variable(torch.FloatTensor(self.batch_size, self.sent_len+1, self.tagset_size).fill_(-10000.)).cuda()
        logalpha[:, 0, self.start_id] = 0.
        istarts = [0] * self.ALLOWED_SPANLEN + range(self.sent_len - self.ALLOWED_SPANLEN+1)
        for i in range(1, self.sent_len+1):
                tmp = scores[:, istarts[i]:i, i-1] + \
                        logalpha[:, istarts[i]:i].unsqueeze(3).expand(self.batch_size, i - istarts[i], self.tagset_size, self.tagset_size)
                tmp = tmp.transpose(1, 3).contiguous().view(self.batch_size, self.tagset_size, (i-istarts[i])*self.tagset_size)
                max_tmp, _ = torch.max(tmp, dim=2)
                tmp = tmp - max_tmp.view(self.batch_size, self.tagset_size, 1)
                logalpha[:, i] = max_tmp + torch.log(torch.sum(torch.exp(tmp), dim=2))

        mask = mask.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.tagset_size)
        alpha = torch.gather(logalpha, 1, mask).squeeze(1)
        return alpha[:,self.stop_id].sum()

    def decode(self, factexprscalars, mask):
        """
        decode SCRF labels with dynamic programming

        args:
            factexprscalars (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : features for SCRF
            mask            (batch_size) : mask for words

        """

        batch_size = factexprscalars.size(0)
        sentlen = factexprscalars.size(1)
        factexprscalars = factexprscalars.data
        logalpha = torch.FloatTensor(batch_size, sentlen+1, self.tagset_size).fill_(-10000.).cuda()
        logalpha[:, 0, self.start_id] = 0.
        starts = torch.zeros((batch_size, sentlen, self.tagset_size)).cuda()
        ys = torch.zeros((batch_size, sentlen, self.tagset_size)).cuda()

        for j in range(1, sentlen + 1):
            istart = 0
            if j > self.ALLOWED_SPANLEN:
                istart = max(0, j - self.ALLOWED_SPANLEN)
            f = factexprscalars[:, istart:j, j - 1].permute(0, 3, 1, 2).contiguous().view(batch_size, self.tagset_size, -1) + \
                logalpha[:, istart:j].contiguous().view(batch_size, 1, -1).expand(batch_size, self.tagset_size, (j - istart) * self.tagset_size)
            logalpha[:, j, :], argm = torch.max(f, dim=2)
            starts[:, j-1, :] = (argm / self.tagset_size + istart)
            ys[:, j-1, :] = (argm % self.tagset_size)

        batch_scores = []
        batch_spans = []
        for i in range(batch_size):
            spans = []
            batch_scores.append(max(logalpha[i, mask[i]-1]))
            end = mask[i]-1
            y = self.stop_id
            while end >= 0:
                start = int(starts[i, end, y])
                y_1 = int(ys[i, end, y])
                spans.append((start, end, y_1, y))
                y = y_1
                end = start - 1
            batch_spans.append(spans)
        return batch_spans, batch_scores

    def get_logloss_numerator(self, goldfactors, scores, mask):
        """
        get scores of best path

        args:
            goldfactors (batch_size, tag_len, 4) : path labels
            scores      (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : all tag scores
            mask        (batch_size, tag_len) : mask for goldfactors

        """
        batch_size = scores.size(0)
        sent_len = scores.size(1)
        tagset_size = scores.size(3)
        goldfactors = goldfactors[:, :, 0]*sent_len*tagset_size*tagset_size + goldfactors[:,:,1]*tagset_size*tagset_size+goldfactors[:,:,2]*tagset_size+goldfactors[:,:,3]
        factorexprs = scores.view(batch_size, -1)
        val = torch.gather(factorexprs, 1, goldfactors)
        numerator = val.masked_select(mask)
        return numerator

    def grConv_scores(self, feats):
        """
        calculate SCRF scores with grConv

        args:
            feats (batch_size, sentence_len, featsdim) : word representations

        """

        scores = Variable(torch.zeros(self.batch_size, self.sent_len, self.sent_len, self.SCRF_feature_dim)).cuda()
        diag0 = torch.LongTensor(range(self.sent_len)).cuda()
        ht = feats
        scores[:, diag0, diag0] = ht
        if self.sent_len == 1:
            return self.toSCRF(scores).unsqueeze(3) + self.transition.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        for span_len in range(1, min(self.ALLOWED_SPANLEN, self.sent_len)):
            ht_1_l = ht[:, :-1]
            ht_1_r = ht[:, 1:]
            h_t_hat = 4 * nn.functional.sigmoid(self.Wl(ht_1_l) + self.Wr(ht_1_r)) - 2
            w = torch.exp(self.Gl(ht_1_l) + self.Gr(ht_1_r)).view(self.batch_size, self.sent_len-span_len, 3, self.SCRF_feature_dim).permute(2,0,1,3)
            w = w / w.sum(0).unsqueeze(0).expand(3, self.batch_size, self.sent_len-span_len, self.SCRF_feature_dim)
            ht = w[0]*h_t_hat + w[1]*ht_1_l + w[2]*ht_1_r
            scores[:, diag0[:-span_len], diag0[span_len:]] = ht

        return self.toSCRF(scores).unsqueeze(3) + self.transition.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    def HSCRF_scores(self, feats):
        ### TODO: need to improve
        """
        calculate SCRF scores with HSCRF

        args:
            feats (batch_size, sentence_len, featsdim) : word representations

        """

        # 3 for O, STOP, START
        validtag_size = self.tagset_size-3
        scores = Variable(torch.zeros(self.batch_size, self.sent_len, self.sent_len, self.tagset_size, self.tagset_size)).cuda()
        diag0 = torch.LongTensor(range(self.sent_len)).cuda()
        # m10000 for STOP
        m10000 = Variable(torch.FloatTensor([-10000.]).expand(self.batch_size, self.sent_len, self.tagset_size, 1)).cuda()
        # m30000 for STOP, START, O
        m30000 = Variable(torch.FloatTensor([-10000.]).expand(self.batch_size, self.sent_len, self.tagset_size, 3)).cuda()
        for span_len in range(min(self.ALLOWED_SPANLEN, self.sent_len)):
            emb_x = self.concat_features(feats, span_len)
            emb_x = self.new_hidden2CRFtag(emb_x)
            if span_len == 0:
                tmp = torch.cat((self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0) + emb_x[:, 0, :, :validtag_size].unsqueeze(2),
                                 m10000,
                                 self.transition[:, -2:].unsqueeze(0).unsqueeze(0) + emb_x[:, 0, :, -2:].unsqueeze(2)), 3)
                scores[:, diag0, diag0] = tmp
            elif span_len == 1:
                tmp = torch.cat((self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.sent_len-1, self.tagset_size, validtag_size) + \
                                                           (emb_x[:, 0, :, validtag_size:2*validtag_size] +
                                                            emb_x[:, 1, :, 3*validtag_size:4*validtag_size]).unsqueeze(2), m30000[:, 1:]), 3)
                scores[:, diag0[:-1], diag0[1:]] = tmp

            elif span_len == 2:
                tmp = torch.cat((self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.sent_len-2, self.tagset_size, validtag_size) + \
                                                           (emb_x[:, 0, :, validtag_size:2*validtag_size] +
                                                            emb_x[:, 1, :, 2*validtag_size:3*validtag_size] +
                                                            emb_x[:, 2, :, 3*validtag_size:4*validtag_size]).unsqueeze(2), m30000[:, 2:]), 3)
                scores[:, diag0[:-2], diag0[2:]] = tmp

            elif span_len >= 3:
                tmp0 = self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.sent_len-span_len, self.tagset_size, validtag_size) + \
                                                           (emb_x[:, 0, :, validtag_size:2*validtag_size] +
                                                            emb_x[:, 1:span_len, :, 2*validtag_size:3*validtag_size].sum(1) +
                                                            emb_x[:, span_len,:, 3*validtag_size:4*validtag_size]).unsqueeze(2)
                tmp = torch.cat((tmp0, m30000[:, span_len:]), 3)
                scores[:, diag0[:-span_len], diag0[span_len:]] = tmp

        return scores

    def concat_features(self, emb_z, span_len):
        """
        concatenate two features

        args:

            emb_z (batch_size, sentence_len, featsdim) : word representations
            span_len: a number (from 0)

        """

        batch_size = emb_z.size(0)
        sent_len = emb_z.size(1)
        hidden_dim = emb_z.size(2)
        emb_z = emb_z.unsqueeze(1).expand(batch_size, sent_len, sent_len, hidden_dim)
        new_emb_z1 = [emb_z[:, i:i + 1, i:i + span_len + 1] for i in range(sent_len - span_len)]
        new_emb_z1 = torch.cat(new_emb_z1, 1)
        new_emb_z2 = (new_emb_z1[:, :, 0]-new_emb_z1[:, :, span_len]).unsqueeze(2).expand(batch_size, sent_len-span_len, span_len+1, hidden_dim)
        index = Variable(torch.LongTensor(range(span_len+1))).cuda()
        index = self.index_embeds(index).unsqueeze(0).unsqueeze(0).expand(batch_size, sent_len-span_len, span_len+1, self.index_embeds_dim)
        new_emb = torch.cat((new_emb_z1, new_emb_z2, index), 3).transpose(1,2).contiguous()

        return new_emb

    def forward(self, feats, mask_word, tags, mask_tag):
        """
        calculate loss

        args:
            feats (batch_size, sent_len, featsdim) : word representations
            mask_word (batch_size) : sentence lengths
            tags (batch_size, tag_len, 4) : target
            mask_tag (batch_size, tag_len) : tag_len <= sentence_len

        """

        self.batch_size = feats.size(0)
        self.sent_len = feats.size(1)
        feats = self.dense(feats)
        if self.grconv:
            self.SCRF_scores = self.grConv_scores(feats)
        else:
            self.SCRF_scores = self.HSCRF_scores(feats)

        forward_score = self.get_logloss_denominator(self.SCRF_scores, mask_word)
        numerator = self.get_logloss_numerator(tags, self.SCRF_scores, mask_tag)

        return (forward_score - numerator.sum()) / self.batch_size

    def get_scrf_decode(self, feats, mask):
        """
        decode with SCRF

        args:
            feats (batch_size, sent_len, featsdim) : word representations
            mask  (batch_size) : mask for words

        """
        self.batch_size = feats.size(0)
        self.sent_len = feats.size(1)
        feats = self.dense(feats)
        if self.grconv:
            self.SCRF_scores = self.grConv_scores(feats)
        else:
            self.SCRF_scores = self.HSCRF_scores(feats)
        batch_spans, batch_scores = self.decode(self.SCRF_scores, mask)
        batch_answer = self.tuple_to_seq(batch_spans)
        return batch_answer, np.array(batch_scores)

    def tuple_to_seq(self, batch_spans):
        batch_answer = []
        for spans in batch_spans:
            answer = utils.tuple_to_seq_BIOES(spans, self.ix_to_tag)
            batch_answer.append(answer[:-1])
        return batch_answer
