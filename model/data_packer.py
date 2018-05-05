from torch.autograd import Variable
from torch.utils.data import Dataset


class Repack:
    """
    Packer for data

    """

    def __init__(self):
        pass

    def repack(self, f_f, f_p, b_f, b_p, w_f, target, mask, len_b, SCRF_labels, mask_SCRF_laebls, cnn_features, test=False):
        """
        packing data

        args:
            f_f              (Batch_size, Char_Seq_len) : forward_char input feature
            f_p              (Batch_size, Word_Seq_len) : forward_char input position
            b_f              (Batch_size, Char_Seq_len) : backward_char input feature
            b_p              (Batch_size, Word_Seq_len) : backward_char input position
            w_f              (Batch_size, Word_Seq_len) : input word feature
            target           (Batch_size, Seq_len) : output target
            mask             (Batch_size, Word_Seq_len) : padding mask
            len_b            (Batch_size, 3) : length of instances in one batch
            SCRF_labels      (Batch_size, Word_Seq_len, 4)  : Semi-CRF labels
            mask_SCRF_laebls (Batch_size, Word_Seq_len) : mask of Semi-CRF labels
            cnn_features     (Batch_size, Word_Seq_len, Word_len): characters features for cnn

        return:
            f_f (Char_Reduced_Seq_len, Batch_size),
            f_p (Word_Reduced_Seq_len, Batch_size),
            b_f (Char_Reduced_Seq_len, Batch_size),
            b_p (Word_Reduced_Seq_len, Batch_size),
            w_f (size Word_Seq_Len, Batch_size),
            target (Reduced_Seq_len, Batch_size),
            mask  (Word_Reduced_Seq_len, Batch_size)
            SCRF_labels (Batch_size, Word_Reduced_Seq_len, 4)
            mask_SCRF_laebls (Batch_size, Word_Reduced_Seq_len)
            cnn_features     (Batch_size, Word_Reduced_Seq_len, word_len)

        """
        mlen, _ = len_b.max(0)
        mlen = mlen.squeeze()
        ocl = b_f.size(1)

        if test:
            f_f = Variable(f_f[:, 0:mlen[0]].transpose(0, 1), volatile=True).cuda()
            f_p = Variable(f_p[:, 0:mlen[1]].transpose(0, 1), volatile=True).cuda()
            b_f = Variable(b_f[:, -mlen[0]:].transpose(0, 1), volatile=True).cuda()
            b_p = Variable((b_p[:, 0:mlen[1]] - ocl + mlen[0]).transpose(0, 1), volatile=True).cuda()
            w_f = Variable(w_f[:, 0:mlen[1]].transpose(0, 1), volatile=True).cuda()
            tg_v = Variable(target[:, 0:mlen[1]].transpose(0, 1), volatile=True).unsqueeze(2).cuda()
            mask_v = Variable(mask[:, 0:mlen[1]].transpose(0, 1), volatile=True).cuda()
            SCRF_labels = Variable(SCRF_labels[:, 0:mlen[2]], volatile=True).cuda()
            mask_SCRF_laebls = Variable(mask_SCRF_laebls[:, 0:mlen[2]], volatile=True).cuda()
            cnn_features = Variable(cnn_features[:, 0:mlen[1], 0:mlen[3]].transpose(0, 1), volatile=True).cuda().contiguous()
        else:
            f_f = Variable(f_f[:, 0:mlen[0]].transpose(0, 1)).cuda()
            f_p = Variable(f_p[:, 0:mlen[1]].transpose(0, 1)).cuda()
            b_f = Variable(b_f[:, -mlen[0]:].transpose(0, 1)).cuda()
            b_p = Variable((b_p[:, 0:mlen[1]] - ocl + mlen[0]).transpose(0, 1)).cuda()
            w_f = Variable(w_f[:, 0:mlen[1]].transpose(0, 1)).cuda()
            tg_v = Variable(target[:, 0:mlen[1]].transpose(0, 1)).unsqueeze(2).cuda()
            mask_v = Variable(mask[:, 0:mlen[1]].transpose(0, 1)).cuda()
            SCRF_labels = Variable(SCRF_labels[:, 0:mlen[2]]).cuda()
            mask_SCRF_laebls = Variable(mask_SCRF_laebls[:, 0:mlen[2]]).cuda()
            cnn_features = Variable(cnn_features[:, 0:mlen[1], 0:mlen[3]].transpose(0, 1)).cuda().contiguous()

        return f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, SCRF_labels, mask_SCRF_laebls, cnn_features



class CRFDataset_WC(Dataset):
    """
    Dataset Class for ner model

    """
    def __init__(self, forw_tensor, forw_index, back_tensor, back_index, word_tensor, label_tensor, mask_tensor, len_tensor, SCRFlabels, mask_SCRFlabels, cnn_features):
        assert forw_tensor.size(0) == label_tensor.size(0)
        assert forw_tensor.size(0) == mask_tensor.size(0)
        assert forw_tensor.size(0) == forw_index.size(0)
        assert forw_tensor.size(0) == back_tensor.size(0)
        assert forw_tensor.size(0) == back_index.size(0)
        assert forw_tensor.size(0) == word_tensor.size(0)
        assert forw_tensor.size(0) == len_tensor.size(0)
        assert forw_tensor.size(0) == SCRFlabels.size(0)
        assert forw_tensor.size(0) == mask_SCRFlabels.size(0)
        assert forw_tensor.size(0) == cnn_features.size(0)

        self.forw_tensor = forw_tensor
        self.forw_index = forw_index
        self.back_tensor = back_tensor
        self.back_index = back_index
        self.word_tensor = word_tensor
        self.label_tensor = label_tensor
        self.mask_tensor = mask_tensor
        self.len_tensor = len_tensor
        self.SCRFlabels = SCRFlabels
        self.mask_SCRFlabels = mask_SCRFlabels
        self.cnn_features = cnn_features

    def __getitem__(self, index):
        return self.forw_tensor[index], self.forw_index[index], self.back_tensor[index], self.back_index[index], self.word_tensor[index], self.label_tensor[index], \
               self.mask_tensor[index], self.len_tensor[index], self.SCRFlabels[index], self.mask_SCRFlabels[index], self.cnn_features[index]

    def __len__(self):
        return self.forw_tensor.size(0)



