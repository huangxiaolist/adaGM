import torch
import time
from utils.time_log import time_since
import os
import sys
from utils.string_helper import *


def preprocess_beam_search_result(beam_search_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk, src_str_list):
    batch_size = beam_search_result['batch_size']
    predictions = beam_search_result['predictions']
    scores = beam_search_result['scores']
    attention = beam_search_result['attention']
    assert len(predictions) == batch_size
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, score_n_best, attn_n_best, oov, src_word_list in zip(predictions, scores, attention, oov_lists, src_str_list):
        # attn_n_best: list of tensor with size [trg_len, src_len], len=n_best
        pred_dict = {}
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk, src_word_list, attn)
            sentences_n_best.append(sentence)
        pred_dict['sentences'] = sentences_n_best  # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_dict['scores'] = score_n_best  # a list of zero dim tensor, with len [n_best]
        pred_dict['attention'] = attn_n_best  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
        pred_list.append(pred_dict)
    return pred_list


def predict_with_beam_search(generator, one2many_data_loader, opt, delimiter_word='<sep>'):
    # file for storing the predicted keyphrases
    if opt.pred_file_prefix == "":
        pred_output_file = open(os.path.join(opt.pred_path, "predictions.txt"), "w")
    else:
        pred_output_file = open(os.path.join(opt.pred_path, "%s_predictions.txt" % opt.pred_file_prefix), "w")
    # debug
    interval = 1000
    start_time_prediction = time.time()
    with torch.no_grad():
        start_time = time.time()
        for batch_i, batch in enumerate(one2many_data_loader):
            if (batch_i + 1) % interval == 0:
                print("Batch %d: Time for running beam search on %d batches : %.1f" % (batch_i+1, interval, time_since(start_time)))
                sys.stdout.flush()
                start_time = time.time()
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, _, _, _, _, original_idx_list = batch
            """
            src: [batch_size, max_src_len], a LongTensor containing the word indices of source sentences, , with oov words replaced by unk idx
            src_lens: [batch_size], a list containing the length of src sequences for each batch
            src_mask: [batch, max_src_len], a FloatTensor
            src_oov: [batch, max_src_len], a LongTensor containing the word indices of source sentences, contains the index of oov words (used by copy)
            oov_lists: [batch_size], a list of oov words for each src, 2dlist
            """
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)

            beam_search_result = generator.beam_search(src, src_lens, src_oov, src_mask, oov_lists, opt.word2idx, opt.max_eos_per_output_seq)
            pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists, opt.word2idx[pykp.io.EOS_WORD], opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk, src_str_list)
            # list of {"sentences": [], "scores": [], "attention": []}

            # recover the original order in the dataset
            seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists),
                               key=lambda p: p[0])
            original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists = zip(*seq_pairs)

            # Process every src in the batch
            for src_str, trg_str_list, pred, oov in zip(src_str_list, trg_str_2dlist, pred_list, oov_lists):
                # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
                # pred_seq_list: a list of sequence objects, sorted by scores
                # oov: a list of oov words
                pred_str_list = pred['sentences']  # predicted sentences from a single src, a list of list of word, with len=[beam_size, out_seq_len], does not include the final <EOS>

                all_keyphrase_list = []  # a list of word list contains all the keyphrases in the top max_n sequences decoded by beam search
                for word_list in pred_str_list:
                    all_keyphrase_list += split_word_list_by_delimiter(word_list, delimiter_word)
                #pred_str_list = [word_list for word_list, is_keep in zip(all_keyphrase_list, not_duplicate_mask) if is_keep]
                pred_str_list = all_keyphrase_list

                # output the predicted keyphrases to a file
                pred_print_out = ''
                for word_list_i, word_list in enumerate(pred_str_list):
                    if word_list_i < len(pred_str_list) - 1:
                        pred_print_out += '%s;' % ' '.join(word_list)
                    else:
                        pred_print_out += '%s' % ' '.join(word_list)
                pred_print_out += '\n'
                pred_output_file.write(pred_print_out)

    pred_output_file.close()
    total_time_prediction = time_since(start_time_prediction)
    print("done!")
    print(total_time_prediction)

