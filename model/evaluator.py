from __future__ import division
import numpy as np
import itertools

import utils


class evaluator():
    """
    evaluation class for ner task

    """

    def __init__(self, packer, l_map, SCRF_l_map):

        self.packer = packer
        self.l_map = l_map
        self.SCRF_l_map = SCRF_l_map
        self.r_l_map = utils.revlut(l_map)
        self.SCRF_r_l_map = utils.revlut(SCRF_l_map)

    def reset(self):
        """
        re-set all states

        """
        self.correct_labels_crf = 0
        self.total_labels_crf = 0
        self.gold_count_crf = 0
        self.guess_count_crf = 0
        self.overlap_count_crf = 0

        self.correct_labels_scrf = 0
        self.total_labels_scrf = 0
        self.gold_count_scrf = 0
        self.guess_count_scrf = 0
        self.overlap_count_scrf = 0

        self.correct_labels_jnt = 0
        self.total_labels_jnt = 0
        self.gold_count_jnt = 0
        self.guess_count_jnt = 0
        self.overlap_count_jnt = 0


    def calc_f1_batch(self, target_data, decoded_data_crfs, decode_data_scrfs, decode_data_jnts):
        """
        update statics for f1 score

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth

        """
        for target, decoded_data_crf, decode_data_scrf, decode_data_jnt in zip(target_data, decoded_data_crfs, decode_data_scrfs, decode_data_jnts):

            length = utils.find_length_from_labels(target, self.l_map)
            gold = target[:length]
            decoded_data_crf = decoded_data_crf[:length]
            decode_data_scrf = decode_data_scrf[:length]
            decode_data_jnt = decode_data_jnt[:length]

            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(
                decoded_data_crf, gold)
            self.correct_labels_crf += correct_labels_i
            self.total_labels_crf += total_labels_i
            self.gold_count_crf += gold_count_i
            self.guess_count_crf += guess_count_i
            self.overlap_count_crf += overlap_count_i

            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(
                decode_data_scrf, gold)
            self.correct_labels_scrf += correct_labels_i
            self.total_labels_scrf += total_labels_i
            self.gold_count_scrf += gold_count_i
            self.guess_count_scrf += guess_count_i
            self.overlap_count_scrf += overlap_count_i

            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(
                decode_data_jnt, gold)
            self.correct_labels_jnt += correct_labels_i
            self.total_labels_jnt += total_labels_i
            self.gold_count_jnt += gold_count_i
            self.guess_count_jnt += guess_count_i
            self.overlap_count_jnt += overlap_count_i


    def f1_score(self):
        """
        calculate f1 score batgsed on statics

        """
        if self.guess_count_crf == 0:
            f_crf, precision_crf, recall_crf, accuracy_crf = 0.0, 0.0, 0.0, 0.0
        else:
            precision_crf = self.overlap_count_crf / float(self.guess_count_crf)
            recall_crf = self.overlap_count_crf / float(self.gold_count_crf)
            if precision_crf == 0.0 or recall_crf == 0.0:
                f_crf, precision_crf, recall_crf, accuracy_crf = 0.0, 0.0, 0.0, 0.0
            else:
                f_crf = 2 * (precision_crf * recall_crf) / (precision_crf + recall_crf)
                accuracy_crf = float(self.correct_labels_crf) / self.total_labels_crf

        if self.guess_count_scrf == 0:
            f_scrf, precision_scrf, recall_scrf, accuracy_scrf = 0.0, 0.0, 0.0, 0.0
        else:
            precision_scrf = self.overlap_count_scrf / float(self.guess_count_scrf)
            recall_scrf = self.overlap_count_scrf / float(self.gold_count_scrf)
            if precision_scrf == 0.0 or recall_scrf == 0.0:
                f_scrf, precision_scrf, recall_scrf, accuracy_scrf = 0.0, 0.0, 0.0, 0.0
            else:
                f_scrf = 2 * (precision_scrf * recall_scrf) / (precision_scrf + recall_scrf)
                accuracy_scrf = float(self.correct_labels_scrf) / self.total_labels_scrf

        if self.guess_count_jnt == 0:
            f_jnt, precision_jnt, recall_jnt, accuracy_jnt = 0.0, 0.0, 0.0, 0.0
        else:
            precision_jnt = self.overlap_count_jnt / float(self.guess_count_jnt)
            recall_jnt = self.overlap_count_jnt / float(self.gold_count_jnt)
            if precision_jnt == 0.0 or recall_jnt == 0.0:
                f_jnt, precision_jnt, recall_jnt, accuracy_jnt = 0.0, 0.0, 0.0, 0.0
            else:
                f_jnt = 2 * (precision_jnt * recall_jnt) / (precision_jnt + recall_jnt)
                accuracy_jnt = float(self.correct_labels_jnt) / self.total_labels_jnt

        return f_crf, precision_crf, recall_crf, accuracy_crf, f_scrf, precision_scrf, recall_scrf, accuracy_scrf, f_jnt, precision_jnt, recall_jnt, accuracy_jnt

    def eval_instance(self, best_path, gold):
        """
        update statics for one instance

        args:
            best_path (seq_len): predicted
            gold (seq_len): ground-truth

        """
        total_labels = len(best_path)
        correct_labels = np.sum(np.equal(best_path, gold))

        gold_chunks = utils.iobes_to_spans(gold, self.r_l_map)
        gold_count = len(gold_chunks)

        guess_chunks = utils.iobes_to_spans(best_path, self.r_l_map)
        guess_count = len(guess_chunks)

        overlap_chunks = gold_chunks & guess_chunks
        overlap_count = len(overlap_chunks)

        return correct_labels, total_labels, gold_count, guess_count, overlap_count


    def calc_score(self, ner_model, dataset_loader):
        ### TODO: need to improve
        """
        calculate F1 score for dev and test sets

        args:
            ner_model: ner model
            dataset_loader: loader class for dev/test set
        """

        ner_model.eval()
        self.reset()

        for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v, SCRF_labels, mask_SCRF_labels, cnn_features in itertools.chain.from_iterable(dataset_loader):

            f_f, f_p, b_f, b_p, w_f, _, mask_v, SCRF_labels, mask_SCRF_labels, cnn_features = self.packer.repack(f_f,f_p,b_f,b_p,w_f,tg,mask_v,len_v,SCRF_labels,mask_SCRF_labels,cnn_features, test=True)

            word_representations = ner_model.word_rep(f_f, f_p, b_f, b_p, w_f, cnn_features)
            bat_size = word_representations.size(1)
            tg = tg.numpy() % len(self.l_map)
            decoded_crf, crf_result_scored_by_crf = utils.decode_with_crf(ner_model.crf, word_representations, mask_v,self.l_map)
            decoded_scrf_seq, scrf_result_scored_by_scrf = ner_model.hscrf.get_scrf_decode(word_representations.transpose(0, 1),mask_v.long().sum(0).data)
            decoded_scrf = []
            for i in decoded_scrf_seq:
                decoded_scrf.append(
                    [self.l_map[j] if j in self.l_map else self.l_map['<pad>'] for j in i] + [
                        self.l_map['<pad>']] * (
                            mask_v.size(0) - 1 - len(i)))
            decoded_scrf = np.array(decoded_scrf)
            if (decoded_crf == decoded_scrf).all():
                decoded_jnt = decoded_scrf
            else:
                decoded_jnt = []

                crf_result_scored_by_scrf = utils.rescored_with_scrf(decoded_crf, self.r_l_map, self.SCRF_l_map, ner_model.hscrf)
                scrf_result_scored_by_crf = utils.rescored_with_crf(decoded_scrf, self.l_map, ner_model.crf.crf_scores)

                crfscores = crf_result_scored_by_crf + crf_result_scored_by_scrf
                scrfscores = scrf_result_scored_by_crf + scrf_result_scored_by_scrf

                for i in range(bat_size):
                    if crfscores[i] > scrfscores[i]:
                        decoded_jnt.append(decoded_crf[i])
                    else:
                        decoded_jnt.append(decoded_scrf[i])
                decoded_jnt = np.array(decoded_jnt)

            self.calc_f1_batch(tg, decoded_crf, decoded_scrf, decoded_jnt)

        return self.f1_score()
