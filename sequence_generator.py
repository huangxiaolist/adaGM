"""
Adapted from
OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py
and seq2seq-keyphrase-pytorch: https://github.com/memray/seq2seq-keyphrase-pytorch
"""
import torch
from beam import Beam
from beam import GNMTGlobalScorer

EPS = 1e-8


class SequenceGenerator(object):
    """Class to generate sequences from an image-to-text model."""

    def __init__(self,
                 model,
                 eos_idx,
                 bos_idx,
                 pad_idx,
                 beam_size,
                 threshold,
                 max_sequence_length,
                 copy_attn=False,
                 include_attn_dist=True,
                 length_penalty_factor=0.0,
                 coverage_penalty_factor=0.0,
                 length_penalty='avg',
                 coverage_penalty='none',
                 cuda=True,
                 n_best=None,
                 ignore_when_blocking=[],
                 ):
        """Initializes the generator.

        Args:
          model: recurrent model, with inputs: (input, dec_hidden) and outputs len(vocab) values
          eos_idx: the idx of the <eos> token
          beam_size: Beam size to use when generating sequences.
          max_sequence_length: The maximum sequence length before stopping the search.
          include_attn_dist: include the attention distribution in the sequence obj or not.
        """
        self.model = model
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.threshold = threshold
        if n_best is None:
            self.n_best = self.beam_size
        else:
            self.n_best = n_best
        self.cuda = cuda
        self.max_sequence_length = max_sequence_length
        self.global_scorer = GNMTGlobalScorer(length_penalty_factor, coverage_penalty_factor, coverage_penalty,
                                              length_penalty)
        self.length_penalty_factor = length_penalty_factor
        self.coverage_penalty_factor = coverage_penalty_factor
        self.include_attn_dist = include_attn_dist
        self.coverage_penalty = coverage_penalty
        self.ignore_when_blocking = ignore_when_blocking

    def beam_search(self, src, src_lens, src_oov, src_mask, oov_lists, word2idx, max_eos_per_output_seq=1):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param oov_lists: list of oov words (idx2word) for each batch, len=batch
        :param word2idx: a dictionary
        """
        self.model.eval()
        batch_size = src.size(0)

        # Encoding
        # [batch_size, max_src_len, memory_bank_size], [batch_size, memory_bank_size]
        memory_bank, encoder_final_state = self.model.encoder(src, src_lens, src_mask)

        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

        # Init decoder state
        decoder_init_state = self.model.init_decoder_state(
            encoder_final_state)  # [dec_layers, batch_size, decoder_size]

        # init initial_input to be BOS token
        # Through the first decoding time step results, we will get the real beam size
        real_beam = self.get_real_beam_size(src, src_mask, src_oov, oov_lists, memory_bank, decoder_init_state,self.threshold)

        self.beam_size = real_beam
        self.n_best = real_beam

        # expand memory_bank, src_mask
        memory_bank = memory_bank.repeat(self.beam_size, 1, 1)  # [batch * beam_size, max_src_len, memory_bank_size]
        src_mask = src_mask.repeat(self.beam_size, 1)  # [batch * beam_size, src_seq_len]
        src_oov = src_oov.repeat(self.beam_size, 1)  # [batch * beam_size, src_seq_len]
        decoder_state = decoder_init_state.repeat(1, self.beam_size, 1)  # [dec_layers, batch_size * beam_size, decoder_size]

        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([word2idx[t]
                                for t in self.ignore_when_blocking])

        beam_list = [
            Beam(self.beam_size, n_best=self.n_best, cuda=self.cuda, global_scorer=self.global_scorer, pad=self.pad_idx,
                 eos=self.eos_idx, bos=self.bos_idx, max_eos_per_output_seq=max_eos_per_output_seq,
                 exclusion_tokens=exclusion_tokens) for _ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a):
            return torch.tensor(a, requires_grad=False)

        '''
        Run beam search.
        '''
        for t in range(1, self.max_sequence_length + 1):

            if all((b.done() for b in beam_list)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            # b.get_current_tokens(): [beam_size]
            # torch.stack([ [beam of batch 1], [beam of batch 2], ... ]) -> [batch, beam]
            # after transpose -> [beam, batch]
            # After flatten, it becomes
            # [batch_1_beam_1, batch_2_beam_1,..., batch_N_beam_1, batch_1_beam_2, ..., batch_N_beam_2, ...]
            # this match the dimension of hidden state
            decoder_input = var(torch.stack([b.get_current_tokens() for b in beam_list])
                                .t().contiguous().view(-1))
            # decoder_input: [batch_size * beam_size]
            if t > 1:
                for batch_idx, beam in enumerate(beam_list):
                    self.set_decoder_state_init(decoder_input, decoder_state, batch_idx, decoder_init_state)

            # Turn any copied words to UNKS
            if self.copy_attn:
                decoder_input = decoder_input.masked_fill(
                    decoder_input.gt(self.model.vocab_size - 1), self.model.unk_idx)

            # Convert the generated eos token to bos token, only useful in one2many_mode=2 or one2many_mode=3
            decoder_input = decoder_input.masked_fill(decoder_input == self.eos_idx, self.bos_idx)

            # run one step of decoding
            # [flattened_batch, vocab_size], [dec_layers, flattened_batch, decoder_size], [flattened_batch, memory_bank_size], [flattened_batch, src_len], [flattened_batch, src_len]
            decoder_dist, decoder_state, attn_dist= \
                self.model.decoder(decoder_input, decoder_state, memory_bank, src_mask, max_num_oov, src_oov)

            log_decoder_dist = torch.log(decoder_dist + EPS)

            # Compute a vector of batch x beam word scores
            log_decoder_dist = log_decoder_dist.view(real_beam, batch_size, -1)  # [beam_size, batch_size, vocab_size]
            attn_dist = attn_dist.view(real_beam, batch_size, -1)  # [beam_size, batch_size, src_seq_len]

            # Advance each beam
            for batch_idx, beam in enumerate(beam_list):
                beam.advance(log_decoder_dist[:, batch_idx], attn_dist[:, batch_idx, :src_lens[batch_idx]])
                self.beam_decoder_state_update(batch_idx, beam.get_current_origin(), decoder_state)

        # Extract sentences from beam.
        result_dict = self._from_beam(beam_list)
        result_dict['batch_size'] = batch_size
        return result_dict

    def _from_beam(self, beam_list):
        ret = {"predictions": [], "scores": [], "attention": []}
        for b in beam_list:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            # Collect all the decoded sentences in to hyps (list of list of idx) and attn (list of tensor)
            for i, (times, k) in enumerate(ks[:n_best]):
                # Get the corresponding decoded sentence, and also the attn dist [seq_len, memory_bank_size].
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(
                hyps)  # 3d list of idx (zero dim tensor), with len [batch_size, n_best, output_seq_len]
            ret['scores'].append(scores)  # a 2d list of zero dim tensor, with len [batch_size, n_best]
            ret["attention"].append(
                attn)  # a 2d list of FloatTensor[output sequence length, src_len] , with len [batch_size, n_best]
            # hyp[::-1]: a list of idx (zero dim tensor), with len = output sequence length
            # torch.stack(attn): FloatTensor, with size: [output sequence length, src_len]
        # print(ret['predictions'])
        return ret

    def set_decoder_state_init(self, decoder_input, decoder_state, batch_idx, decoder_init_state):
        decoder_layers, flattened_batch_size, decoder_size = list(decoder_state.size())
        assert flattened_batch_size % self.beam_size == 0
        original_batch_size = flattened_batch_size // self.beam_size
        # select the hidden states of a particular batch, [dec_layers, batch_size * beam_size, decoder_size] -> [dec_layers, beam_size, decoder_size]
        decoder_state_transformed = decoder_state.view(decoder_layers, self.beam_size, original_batch_size,
                                                       decoder_size)[:, :, batch_idx]
        # select the hidden states of the beams specified by the beam_indices -> [dec_layers, beam_size, decoder_size]
        decoder_input_eos = decoder_input == self.eos_idx
        decoder_input_eos = decoder_input_eos.view(self.beam_size, original_batch_size)[:, batch_idx]
        for idx in range(self.beam_size):
            if decoder_input_eos[idx]:
                # [dec_layer, decoder_size]
                decoder_state_transformed[:, idx] = decoder_init_state[:, batch_idx]
        decoder_state_transformed.data.copy_(decoder_state_transformed)

    def beam_decoder_state_update(self, batch_idx, beam_indices, decoder_state):
        """
        :param batch_idx: int
        :param beam_indices: a long tensor of previous beam indices, size: [beam_size]
        :param decoder_state: [dec_layers, flattened_batch_size, decoder_size]
        :return:
        """
        decoder_layers, flattened_batch_size, decoder_size = list(decoder_state.size())
        assert flattened_batch_size % self.beam_size == 0
        original_batch_size = flattened_batch_size // self.beam_size
        # select the hidden states of a particular batch, [dec_layers, batch_size * beam_size, decoder_size] -> [dec_layers, beam_size, decoder_size]
        decoder_state_transformed = decoder_state.view(decoder_layers, self.beam_size, original_batch_size,
                                                       decoder_size)[:, :, batch_idx]
        # select the hidden states of the beams specified by the beam_indices -> [dec_layers, beam_size, decoder_size]
        decoder_state_transformed.data.copy_(decoder_state_transformed.data.index_select(1, beam_indices))

    def get_real_beam_size(self, src, src_mask, src_oov, oov_lists, memory_bank, decoder_init_state, threshold):
        batch_size = src.size(0)

        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
        # expand memory_bank, src_mask
        decoder_state = decoder_init_state
        decoder_input = torch.LongTensor([1]).cuda()

        # run one step of decoding
        decoder_dist, _, _ = \
            self.model.decoder(decoder_input, decoder_state, memory_bank, src_mask, max_num_oov, src_oov)
        best_score, best_scores_idx = decoder_dist.view(batch_size, -1).view(-1).topk(20, 0, True, True)
        real_beam = 0
        for score in best_score:
            if score > threshold:
                real_beam += 1
        return real_beam
