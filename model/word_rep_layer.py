import torch
import torch.nn as nn
import utils
import highway_layer


class WORD_REP(nn.Module):

    def __init__(self, char_size, char_embedding_dim, char_hidden_dim, cnn_filter_num, char_lstm_layers, word_embedding_dim,
                 word_hidden_dim, word_lstm_layers, vocab_size, dropout_ratio, if_highway=False,
                 in_doc_words=2, highway_layers=1, char_lstm=True):

        super(WORD_REP, self).__init__()
        self.char_embedding_dim = char_embedding_dim
        self.char_hidden_dim = char_hidden_dim
        self.cnn_filter_num = cnn_filter_num
        self.char_size = char_size
        self.word_embedding_dim = word_embedding_dim
        self.word_hidden_dim = word_hidden_dim
        self.word_size = vocab_size
        self.char_lstm = char_lstm
        self.if_highway = if_highway

        self.char_embeds = nn.Embedding(char_size, char_embedding_dim)
        self.word_embeds = nn.Embedding(vocab_size, word_embedding_dim)

        if char_lstm:
            self.crit_lm = nn.CrossEntropyLoss()
            self.forw_char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim, num_layers=char_lstm_layers, bidirectional=False,
                                      dropout=dropout_ratio)
            self.back_char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim, num_layers=char_lstm_layers, bidirectional=False,
                                      dropout=dropout_ratio)
            self.word_lstm_lm = nn.LSTM(word_embedding_dim + char_hidden_dim * 2, word_hidden_dim // 2,
                                        num_layers=word_lstm_layers,
                                        bidirectional=True, dropout=dropout_ratio)
            self.char_pre_train_out = nn.Linear(char_hidden_dim, char_size)
            self.word_pre_train_out = nn.Linear(char_hidden_dim, in_doc_words)
            if self.if_highway:
                self.forw2char = highway_layer.hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
                self.back2char = highway_layer.hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
                self.forw2word = highway_layer.hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
                self.back2word = highway_layer.hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
                self.fb2char = highway_layer.hw(2*char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
        else:
            self.cnn = nn.Conv2d(1, cnn_filter_num, (3, char_embedding_dim), padding=(2, 0))

            self.word_lstm_cnn = nn.LSTM(word_embedding_dim + cnn_filter_num, word_hidden_dim // 2, num_layers=word_lstm_layers, bidirectional=True,
                              dropout=dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.batch_size = 1
        self.word_seq_length = 1


    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.word_seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_word_embedding(self, pre_word_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding

        """
        assert (pre_word_embeddings.size()[1] == self.word_embedding_dim)
        self.word_embeds.weight = nn.Parameter(pre_word_embeddings)

    def rand_init(self):
        """
        random initialization

        args:
            init_char_embedding: random initialize char embedding or not

        """

        utils.init_embedding(self.char_embeds.weight)
        if self.char_lstm:
            utils.init_lstm(self.forw_char_lstm)
            utils.init_lstm(self.back_char_lstm)
            utils.init_lstm(self.word_lstm_lm)
            utils.init_linear(self.char_pre_train_out)
            utils.init_linear(self.word_pre_train_out)
            if self.if_highway:
                self.forw2char.rand_init()
                self.back2char.rand_init()
                self.forw2word.rand_init()
                self.back2word.rand_init()
                self.fb2char.rand_init()
        else:
            utils.init_lstm(self.word_lstm_cnn)

    def word_pre_train_forward(self, sentence, position):
        """
        output of forward language model

        args:
            sentence (char_seq_len, batch_size): char-level representation of sentence
            position (word_seq_len, batch_size): position of blank space in char-level representation of sentence

        """

        embeds = self.char_embeds(sentence)
        d_embeds = self.dropout(embeds)
        lstm_out, hidden = self.forw_char_lstm(d_embeds)

        tmpsize = position.size()
        position = position.unsqueeze(2).expand(tmpsize[0], tmpsize[1], self.char_hidden_dim)
        select_lstm_out = torch.gather(lstm_out, 0, position)
        d_lstm_out = self.dropout(select_lstm_out).view(-1, self.char_hidden_dim)

        if self.if_highway:
            char_out = self.forw2word(d_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = d_lstm_out

        pre_score = self.word_pre_train_out(d_char_out)
        return pre_score, hidden

    def word_pre_train_backward(self, sentence, position):
        """
        output of backward language model

        args:
            sentence (char_seq_len, batch_size): char-level representation of sentence (inverse order)
            position (word_seq_len, batch_size): position of blank space in inversed char-level representation of sentence

        """
        embeds = self.char_embeds(sentence)
        d_embeds = self.dropout(embeds)
        lstm_out, hidden = self.back_char_lstm(d_embeds)

        tmpsize = position.size()
        position = position.unsqueeze(2).expand(tmpsize[0], tmpsize[1], self.char_hidden_dim)
        select_lstm_out = torch.gather(lstm_out, 0, position)
        d_lstm_out = self.dropout(select_lstm_out).view(-1, self.char_hidden_dim)

        if self.if_highway:
            char_out = self.back2word(d_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = d_lstm_out

        pre_score = self.word_pre_train_out(d_char_out)
        return pre_score, hidden

    def lm_loss(self, f_f, f_p, b_f, b_p, w_f):
        """
        language model loss

        args:
            f_f (char_seq_len, batch_size) : char-level representation of sentence
            f_p (word_seq_len, batch_size) : position of blank space in char-level representation of sentence
            b_f (char_seq_len, batch_size) : char-level representation of sentence (inverse order)
            b_p (word_seq_len, batch_size) : position of blank space in inversed char-level representation of sentence
            w_f (word_seq_len, batch_size) : word-level representation of sentence

        """

        cf_p = f_p[0:-1, :].contiguous()
        cb_p = b_p[1:, :].contiguous()
        cf_y = w_f[1:, :].contiguous()
        cb_y = w_f[0:-1, :].contiguous()
        cfs, _ = self.word_pre_train_forward(f_f, cf_p)
        loss = self.crit_lm(cfs, cf_y.view(-1))
        cbs, _ = self.word_pre_train_backward(b_f, cb_p)
        loss = loss + self.crit_lm(cbs, cb_y.view(-1))

        return loss

    def cnn_lstm(self, word_seq, cnn_features):
        """
        return word representations with character-cnn

        args:
            word_seq:     word_seq_len, batch_size
            cnn_features: word_seq_len, batch_size, word_len

        """
        self.set_batch_seq_size(word_seq)
        cnn_features = cnn_features.view(cnn_features.size(0) * cnn_features.size(1), -1)
        cnn_features = self.char_embeds(cnn_features).view(cnn_features.size(0), 1, cnn_features.size(1), -1)
        cnn_features = self.cnn(cnn_features)
        d_char_out = nn.functional.max_pool2d(cnn_features,
                                              kernel_size=(cnn_features.size(2), 1)).view(self.word_seq_length, self.batch_size, self.cnn_filter_num)
        word_emb = self.word_embeds(word_seq)

        word_input = torch.cat((word_emb, d_char_out), dim=2)
        word_input = self.dropout(word_input)

        lstm_out, _ = self.word_lstm_cnn(word_input)
        lstm_out = self.dropout(lstm_out)

        return lstm_out

    def lm_lstm(self, forw_sentence, forw_position, back_sentence, back_position, word_seq):
        '''
        return word representations with character-language-model

        args:
            forw_sentence (char_seq_len, batch_size) : char-level representation of sentence
            forw_position (word_seq_len, batch_size) : position of blank space in char-level representation of sentence
            back_sentence (char_seq_len, batch_size) : char-level representation of sentence (inverse order)
            back_position (word_seq_len, batch_size) : position of blank space in inversed char-level representation of sentence
            word_seq (word_seq_len, batch_size) : word-level representation of sentence

        '''

        self.set_batch_seq_size(forw_position)

        forw_emb = self.char_embeds(forw_sentence)
        back_emb = self.char_embeds(back_sentence)

        d_f_emb = self.dropout(forw_emb)
        d_b_emb = self.dropout(back_emb)

        forw_lstm_out, _ = self.forw_char_lstm(d_f_emb)

        back_lstm_out, _ = self.back_char_lstm(d_b_emb)

        forw_position = forw_position.unsqueeze(2).expand(self.word_seq_length, self.batch_size, self.char_hidden_dim)
        select_forw_lstm_out = torch.gather(forw_lstm_out, 0, forw_position)

        back_position = back_position.unsqueeze(2).expand(self.word_seq_length, self.batch_size, self.char_hidden_dim)
        select_back_lstm_out = torch.gather(back_lstm_out, 0, back_position)

        fb_lstm_out = self.dropout(torch.cat((select_forw_lstm_out, select_back_lstm_out), dim=2))
        if self.if_highway:
            char_out = self.fb2char(fb_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = fb_lstm_out

        word_emb = self.word_embeds(word_seq)
        d_word_emb = self.dropout(word_emb)

        word_input = torch.cat((d_word_emb, d_char_out), dim=2)

        lstm_out, _ = self.word_lstm_lm(word_input)
        d_lstm_out = self.dropout(lstm_out)

        return d_lstm_out

    def forward(self, forw_sentence, forw_position, back_sentence, back_position, word_seq, cnn_features):
        '''
        word representations

        args:
            forw_sentence (char_seq_len, batch_size) : char-level representation of sentence
            forw_position (word_seq_len, batch_size) : position of blank space in char-level representation of sentence
            back_sentence (char_seq_len, batch_size) : char-level representation of sentence (inverse order)
            back_position (word_seq_len, batch_size) : position of blank space in inversed char-level representation of sentence
            word_seq (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        '''
        if self.char_lstm:
            return self.lm_lstm(forw_sentence, forw_position, back_sentence, back_position, word_seq)
        else:
            return self.cnn_lstm(word_seq, cnn_features)
