import pykp
from pykp.rnn_encoder import *
from pykp.rnn_decoder import RNNDecoder


class Seq2SeqModel(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqModel, self).__init__()

        self.vocab_size = opt.vocab_size
        self.emb_dim = opt.word_vec_size
        self.num_directions = 2 if opt.bidirectional else 1
        self.encoder_size = opt.encoder_size
        self.decoder_size = opt.decoder_size

        self.batch_size = opt.batch_size
        self.bidirectional = opt.bidirectional
        self.enc_layers = opt.enc_layers
        self.dec_layers = opt.dec_layers
        self.dropout = opt.dropout

        self.bridge = opt.bridge

        self.one2many = opt.one2many

        self.copy_attn = opt.copy_attention

        self.pad_idx_src = opt.word2idx[pykp.io.PAD_WORD]
        self.pad_idx_trg = opt.word2idx[pykp.io.PAD_WORD]
        self.bos_idx = opt.word2idx[pykp.io.BOS_WORD]
        self.eos_idx = opt.word2idx[pykp.io.EOS_WORD]
        self.unk_idx = opt.word2idx[pykp.io.UNK_WORD]
        self.sep_idx = opt.word2idx[pykp.io.SEP_WORD]

        self.share_embeddings = opt.share_embeddings

        self.attn_mode = opt.attn_mode

        self.encoder = RNNEncoderBasic(
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.encoder_size,
            num_layers=self.enc_layers,
            bidirectional=self.bidirectional,
            pad_token=self.pad_idx_src,
            dropout=self.dropout
        )

        self.decoder = RNNDecoder(
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.decoder_size,
            num_layers=self.dec_layers,
            memory_bank_size=self.num_directions * self.encoder_size,
            copy_attn=self.copy_attn,
            pad_idx=self.pad_idx_trg,
            attn_mode=self.attn_mode,
            dropout=self.dropout
        )

        if self.bridge == 'dense':
            self.bridge_layer = nn.Linear(self.encoder_size * self.num_directions, self.decoder_size)
        elif opt.bridge == 'dense_nonlinear':
            self.bridge_layer = nn.tanh(nn.Linear(self.encoder_size * self.num_directions, self.decoder_size))
        else:
            self.bridge_layer = None

        if self.bridge == 'copy':
            assert self.encoder_size * self.num_directions == self.decoder_size, 'encoder hidden size and decoder hidden size are not match, please use a bridge layer'

        if self.share_embeddings:
            self.encoder.embedding.weight = self.decoder.embedding.weight

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)
        if not self.share_embeddings:
            self.decoder.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_lens, trg, src_oov, max_num_oov, src_mask, num_trgs=None):
        """
        Args:
            src: [batch_size, max_src_len], a LongTensor containing the word indices of source sentences, with oov words replaced by unk idx
            src_lens: [batch_size], a list containing the length of src sequences for each batch, with oov words replaced by unk idx
            trg: [batch_size, max_trg_len], a LongTensor containing the word indices of target sentences
            src_oov: [batch_size, max_src_len], a LongTensor containing the word indices of source sentences, contains the index of oov words (used by copy)
            max_num_oov: int, max number of oov for each batch
            src_mask: [batch, max_src_len], a FloatTensor
            num_trgs: [batch_size], only effective in one2many mode, a list of num of targets in each batch
        Returns:
            decoder_dist_all:[batch_size,max_trg_len, vocab_size]
            attention_dist_all:[batch_size, max_trg_len, max_src_len]
        """
        batch_size, max_src_len = list(src.size())

        # Encoding
        memory_bank, encoder_final_state = self.encoder(src, src_lens, src_mask)
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])

        # Decoding
        h_t_init = self.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]
        max_target_length = trg.size(1)

        decoder_dist_all = []
        attention_dist_all = []

        y_t_init = trg.new_ones(batch_size) * self.bos_idx  # [batch_size]
        y_t_next, pred_counters, re_init_indicators = None, None, None
        h_t_next = None
        for t in range(max_target_length):
            # determine the hidden state that will be feed into the next step
            # according to the time step or the target input
            #re_init_indicators = (y_t == self.eos_idx)  # [batch]
            if t == 0:
                pred_counters = trg.new_zeros(batch_size, dtype=torch.uint8)  # [batch_size]
            else:
                re_init_indicators = (y_t_next == self.eos_idx)  # [batch_size]
                pred_counters += re_init_indicators

            if t == 0:
                h_t = h_t_init
                y_t = y_t_init

            elif self.one2many and re_init_indicators.sum().item() > 0:

                h_t = []
                y_t = []
                # h_t_next [dec_layers, batch_size, decoder_size]
                # h_t_init [dec_layers, batch_size, decoder_size]
                for batch_idx, (indicator, pred_count, trg_count) in enumerate(zip(re_init_indicators, pred_counters, num_trgs)):
                    if indicator.item() == 1 and pred_count.item() < trg_count:
                        # Reset State Mechanism
                        h_t.append(h_t_init[:, batch_idx, :].unsqueeze(1))
                        y_t.append(y_t_init[batch_idx].unsqueeze(0))
                    else:  # indicator.item() == 0 or (indicator.item() == 1 and pred_count.item() == trg_count):
                        h_t.append(h_t_next[:, batch_idx, :].unsqueeze(1))
                        y_t.append(y_t_next[batch_idx].unsqueeze(0))
                h_t = torch.cat(h_t, dim=1)  # [dec_layers, batch_size, decoder_size]
                y_t = torch.cat(y_t, dim=0)  # [batch_size]
            else:
                # regular seq2seq model
                h_t = h_t_next
                y_t = y_t_next

            decoder_dist, h_t_next, attn_dist = \
                self.decoder(y_t, h_t, memory_bank, src_mask, max_num_oov, src_oov)
            decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
            attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]

            y_t_next = trg[:, t]  # [batch]

        decoder_dist_all = torch.cat(decoder_dist_all, dim=1)  # [batch_size, trg_len, vocab_size]
        attention_dist_all = torch.cat(attention_dist_all, dim=1)  # [batch_size, trg_len, src_len]

        if self.copy_attn:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size + max_num_oov))
        else:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size))
        assert attention_dist_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

        return decoder_dist_all, attention_dist_all

    def init_decoder_state(self, encoder_final_state):
        """
        Args:
            encoder_final_state: [batch_size, num_directions * encoder_size]
        Returns:
            decoder_init_state: [dec_layers, batch_size, decoder_size]
        """
        batch_size = encoder_final_state.size(0)
        if self.bridge == 'none':
            decoder_init_state = None
        elif self.bridge == 'copy':
            decoder_init_state = encoder_final_state
        else:
            decoder_init_state = self.bridge_layer(encoder_final_state)
        decoder_init_state = decoder_init_state.unsqueeze(0).expand((self.dec_layers, batch_size, self.decoder_size))
        # [dec_layers, batch_size, decoder_size]
        return decoder_init_state