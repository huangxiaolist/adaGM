"""
Calc the F1@5、F1@M、DupRatio、MAE
"""
import numpy as np
import argparse
import config
from utils.string_helper import *
from collections import defaultdict
import os
import pykp.io
import pickle


def check_valid_keyphrases(str_list):
    num_pred_seq = len(str_list)
    is_valid = np.zeros(num_pred_seq, dtype=bool)
    for i, word_list in enumerate(str_list):
        keep_flag = True

        if len(word_list) == 0:
            keep_flag = False

        for w in word_list:
            if opt.invalidate_unk:
                if w == pykp.io.UNK_WORD or w == ',' or w == '.':
                    keep_flag = False
            else:
                if w == ',' or w == '.':
                    keep_flag = False
        is_valid[i] = keep_flag

    return is_valid


def compute_extra_one_word_seqs_mask(str_list):
    num_pred_seq = len(str_list)
    mask = np.zeros(num_pred_seq, dtype=bool)
    num_one_word_seqs = 0
    for i, word_list in enumerate(str_list):
        if len(word_list) == 1:
            num_one_word_seqs += 1
            if num_one_word_seqs > 1:
                mask[i] = False
                continue
        mask[i] = True
    return mask, num_one_word_seqs


def check_duplicate_keyphrases(keyphrase_str_list):
    """
    :param keyphrase_str_list: a 2d list of tokens
    :return: a boolean np array indicate, 1 = unique, 0 = duplicate
    """
    num_keyphrases = len(keyphrase_str_list)
    not_duplicate = np.ones(num_keyphrases, dtype=bool)
    keyphrase_set = set()
    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        if '_'.join(keyphrase_word_list) in keyphrase_set:
            not_duplicate[i] = False
        else:
            not_duplicate[i] = True
        keyphrase_set.add('_'.join(keyphrase_word_list))
    return not_duplicate


def check_present_keyphrases(src_str, keyphrase_str_list, match_by_str=False):
    """

    Args:
        src_str: stemmed word list of source text
        keyphrase_str_list: stemmed list of word list
        match_by_str:

    Returns:

    """
    num_keyphrases = len(keyphrase_str_list)
    is_present = np.zeros(num_keyphrases, dtype=bool)

    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        joined_keyphrase_str = ' '.join(keyphrase_word_list)

        if joined_keyphrase_str.strip() == "":  # if the keyphrase is an empty string
            is_present[i] = False
        else:
            if not match_by_str:  # match by word
                # check if it appears in source text
                match = False
                for src_start_idx in range(len(src_str) - len(keyphrase_word_list) + 1):
                    match = True
                    for keyphrase_i, keyphrase_w in enumerate(keyphrase_word_list):
                        src_w = src_str[src_start_idx + keyphrase_i]
                        if src_w != keyphrase_w:
                            match = False
                            break
                    if match:
                        break
                if match:
                    is_present[i] = True
                else:
                    is_present[i] = False
            else:  # match by str
                if joined_keyphrase_str in ' '.join(src_str):
                    is_present[i] = True
                else:
                    is_present[i] = False
    return is_present


def compute_match_result(trg_str_list, pred_str_list, type_='exact', dimension=1):
    assert type_ in ['exact', 'sub'], "Right now only support exact matching and substring matching"
    assert dimension in [1, 2], "only support 1 or 2"
    num_pred_str = len(pred_str_list)
    num_trg_str = len(trg_str_list)
    if dimension == 1:
        is_match = np.zeros(num_pred_str, dtype=bool)
        for pred_idx, pred_word_list in enumerate(pred_str_list):
            joined_pred_word_list = ' '.join(pred_word_list)
            for trg_idx, trg_word_list in enumerate(trg_str_list):
                joined_trg_word_list = ' '.join(trg_word_list)
                if type_ == 'exact':
                    if joined_pred_word_list == joined_trg_word_list:
                        is_match[pred_idx] = True
                        break
                elif type_ == 'sub':
                    if joined_pred_word_list in joined_trg_word_list:
                        is_match[pred_idx] = True
                        break
    else:
        is_match = np.zeros((num_trg_str, num_pred_str), dtype=bool)
        for trg_idx, trg_word_list in enumerate(trg_str_list):
            joined_trg_word_list = ' '.join(trg_word_list)
            for pred_idx, pred_word_list in enumerate(pred_str_list):
                joined_pred_word_list = ' '.join(pred_word_list)
                if type_ == 'exact':
                    if joined_pred_word_list == joined_trg_word_list:
                        is_match[trg_idx][pred_idx] = True
                elif type_ == 'sub':
                    if joined_pred_word_list in joined_trg_word_list:
                        is_match[trg_idx][pred_idx] = True
    return is_match


def compute_classification_metrics_at_k(is_match, num_predictions, num_trgs, topk=5, meng_rui_precision=False):
    """

    Args:
        is_match: a boolean np array with size [num_predictions]
        num_predictions:
        num_trgs:
        topk:
        meng_rui_precision:

    Returns:
        {'precision@%d' % topk: precision_k, 'recall@%d' % topk: recall_k, 'f1_score@%d' % topk: f1,
        'num_matches@%d': num_matches}
    """
    assert is_match.shape[0] == num_predictions
    if topk == 'M':
        topk = num_predictions
    elif topk == 'G':
        # topk = num_trgs
        if num_predictions < num_trgs:
            topk = num_trgs
        else:
            topk = num_predictions

    if meng_rui_precision:
        if num_predictions > topk:
            is_match = is_match[:topk]
            num_predictions_k = topk
        else:
            num_predictions_k = num_predictions
    else:
        if num_predictions > topk:
            is_match = is_match[:topk]
        num_predictions_k = topk

    num_matches_k = sum(is_match)

    precision_k, recall_k, f1_k = compute_classification_metrics(num_matches_k, num_predictions_k, num_trgs)

    return precision_k, recall_k, f1_k, num_matches_k, num_predictions_k


def compute_classification_metrics_at_ks(is_match, num_predictions, num_trgs, k_list=[5, 10], meng_rui_precision=False):
    """

    Args:
        is_match: a boolean np array with size [num_predictions]
        num_predictions:
        num_trgs:
        k_list:
        meng_rui_precision:

    Returns:
        {'precision@%d' % topk: precision_k, 'recall@%d' % topk: recall_k, 'f1_score@%d' % topk: f1,
        'num_matches@%d': num_matches}
    """
    assert is_match.shape[0] == num_predictions
    # topk.sort()
    if num_predictions == 0:
        precision_ks = [0] * len(k_list)
        recall_ks = [0] * len(k_list)
        f1_ks = [0] * len(k_list)
        num_matches_ks = [0] * len(k_list)
        num_predictions_ks = [0] * len(k_list)
    else:
        num_matches = np.cumsum(is_match)
        num_predictions_ks = []
        num_matches_ks = []
        precision_ks = []
        recall_ks = []
        f1_ks = []
        for topk in k_list:
            if topk == 'M':
                topk = num_predictions
            elif topk == 'G':
                # topk = num_trgs
                if num_predictions < num_trgs:
                    topk = num_trgs
                else:
                    topk = num_predictions

            if meng_rui_precision:
                if num_predictions > topk:
                    num_matches_at_k = num_matches[topk - 1]
                    num_predictions_at_k = topk
                else:
                    num_matches_at_k = num_matches[-1]
                    num_predictions_at_k = num_predictions
            else:
                if num_predictions > topk:
                    num_matches_at_k = num_matches[topk - 1]
                else:
                    num_matches_at_k = num_matches[-1]
                num_predictions_at_k = topk

            precision_k, recall_k, f1_k = compute_classification_metrics(num_matches_at_k, num_predictions_at_k,
                                                                         num_trgs)
            precision_ks.append(precision_k)
            recall_ks.append(recall_k)
            f1_ks.append(f1_k)
            num_matches_ks.append(num_matches_at_k)
            num_predictions_ks.append(num_predictions_at_k)
    return precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks


def compute_classification_metrics(num_matches, num_predictions, num_trgs):
    precision = compute_precision(num_matches, num_predictions)
    recall = compute_recall(num_matches, num_trgs)
    f1 = compute_f1(precision, recall)
    return precision, recall, f1


def compute_precision(num_matches, num_predictions):
    return num_matches / num_predictions if num_predictions > 0 else 0.0


def compute_recall(num_matches, num_trgs):
    return num_matches / num_trgs if num_trgs > 0 else 0.0


def compute_f1(precision, recall):
    return float(2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0.0


def average_precision(r, num_predictions, num_trgs):
    if num_predictions == 0 or num_trgs == 0:
        return 0
    r_cum_sum = np.cumsum(r, axis=0)
    precision_sum = sum([compute_precision(r_cum_sum[k], k + 1) for k in range(num_predictions) if r[k]])
    '''
    precision_sum = 0
    for k in range(num_predictions):
        if r[k] is False:
            continue
        else:
            precision_k = precision(r_cum_sum[k], k+1)
            precision_sum += precision_k
    '''
    return precision_sum / num_trgs


def average_precision_at_k(r, k, num_predictions, num_trgs):
    if k == 'M':
        k = num_predictions
    elif k == 'G':
        # k = num_trgs
        if num_predictions < num_trgs:
            k = num_trgs
        else:
            k = num_predictions

    if k < num_predictions:
        num_predictions = k
        r = r[:k]
    return average_precision(r, num_predictions, num_trgs)


def average_precision_at_ks(r, k_list, num_predictions, num_trgs):
    if num_predictions == 0 or num_trgs == 0:
        return [0] * len(k_list)
    # k_max = max(k_list)
    k_max = -1
    for k in k_list:
        if k == 'M':
            k = num_predictions
        elif k == 'G':
            # k = num_trgs
            if num_predictions < num_trgs:
                k = num_trgs
            else:
                k = num_predictions
        if k > k_max:
            k_max = k
    if num_predictions > k_max:
        num_predictions = k_max
        r = r[:num_predictions]
    r_cum_sum = np.cumsum(r, axis=0)
    precision_array = [compute_precision(r_cum_sum[k], k + 1) * r[k] for k in range(num_predictions)]
    precision_cum_sum = np.cumsum(precision_array, axis=0)
    average_precision_array = precision_cum_sum / num_trgs
    return_indices = []
    for k in k_list:
        if k == 'M':
            k = num_predictions
        elif k == 'G':
            # k = num_trgs
            if num_predictions < num_trgs:
                k = num_trgs
            else:
                k = num_predictions
        return_indices.append((k - 1) if k <= num_predictions else (num_predictions - 1))
    return_indices = np.array(return_indices, dtype=int)
    return average_precision_array[return_indices]


def update_score_dict(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed, k_list, score_dict, tag):
    num_targets = len(trg_token_2dlist_stemmed)
    num_predictions = len(pred_token_2dlist_stemmed)

    is_match = compute_match_result(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed,
                                    type_='exact', dimension=1)
    # Classification metrics
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
        compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list,
                                             meng_rui_precision=opt.meng_rui_precision)

    # Ranking metrics

    ap_ks = average_precision_at_ks(is_match, k_list=k_list,
                                    num_predictions=num_predictions, num_trgs=num_targets)

    for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k, ap_k in \
            zip(k_list, precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks, ap_ks):
        score_dict['precision@{}_{}'.format(topk, tag)].append(precision_k)
        score_dict['recall@{}_{}'.format(topk, tag)].append(recall_k)
        score_dict['f1_score@{}_{}'.format(topk, tag)].append(f1_k)
        score_dict['num_matches@{}_{}'.format(topk, tag)].append(num_matches_k)
        score_dict['num_predictions@{}_{}'.format(topk, tag)].append(num_predictions_k)
        score_dict['num_targets@{}_{}'.format(topk, tag)].append(num_targets)
        score_dict['AP@{}_{}'.format(topk, tag)].append(ap_k)

    score_dict['num_targets_{}'.format(tag)].append(num_targets)
    score_dict['num_predictions_{}'.format(tag)].append(num_predictions)
    return score_dict


def filter_prediction(disable_valid_filter, disable_extra_one_word_filter, pred_token_2dlist_stemmed):
    """
    Remove the duplicate predictions, can optionally remove invalid predictions and extra one word predictions
    :param disable_valid_filter:
    :param disable_extra_one_word_filter:
    :param pred_token_2dlist_stemmed:
    :return:
    """
    num_predictions = len(pred_token_2dlist_stemmed)
    is_unique_mask = check_duplicate_keyphrases(pred_token_2dlist_stemmed)  # boolean array, 1=unqiue, 0=duplicate
    pred_filter = is_unique_mask
    if not disable_valid_filter:
        is_valid_mask = check_valid_keyphrases(pred_token_2dlist_stemmed)
        pred_filter = pred_filter * is_valid_mask
    if not disable_extra_one_word_filter:
        extra_one_word_seqs_mask, num_one_word_seqs = compute_extra_one_word_seqs_mask(pred_token_2dlist_stemmed)
        pred_filter = pred_filter * extra_one_word_seqs_mask
    filtered_stemmed_pred_str_list = [word_list for word_list, is_keep in
                                      zip(pred_token_2dlist_stemmed, pred_filter) if
                                      is_keep]
    num_duplicated_predictions = num_predictions - np.sum(is_unique_mask)
    return filtered_stemmed_pred_str_list, num_duplicated_predictions


def find_unique_target(trg_token_2dlist_stemmed):
    """
    Remove the duplicate targets
    :param trg_token_2dlist_stemmed:
    :return:
    """
    num_trg = len(trg_token_2dlist_stemmed)
    is_unique_mask = check_duplicate_keyphrases(trg_token_2dlist_stemmed)  # boolean array, 1=unqiue, 0=duplicate
    trg_filter = is_unique_mask
    filtered_stemmed_trg_str_list = [word_list for word_list, is_keep in
                                     zip(trg_token_2dlist_stemmed, trg_filter) if
                                     is_keep]
    num_duplicated_trg = num_trg - np.sum(is_unique_mask)
    return filtered_stemmed_trg_str_list, num_duplicated_trg


def separate_present_absent_by_source(src_token_list_stemmed, keyphrase_token_2dlist_stemmed, match_by_str):
    is_present_mask = check_present_keyphrases(src_token_list_stemmed, keyphrase_token_2dlist_stemmed, match_by_str)
    present_keyphrase_token2dlist = []
    absent_keyphrase_token2dlist = []
    for keyphrase_token_list, is_present in zip(keyphrase_token_2dlist_stemmed, is_present_mask):
        if is_present:
            present_keyphrase_token2dlist.append(keyphrase_token_list)
        else:
            absent_keyphrase_token2dlist.append(keyphrase_token_list)
    return present_keyphrase_token2dlist, absent_keyphrase_token2dlist


def process_input_ks(ks):
    ks_list = []
    for k in ks:
        if k != 'M' and k != 'G':
            k = int(k)
        ks_list.append(k)
    return ks_list


def main(opt):
    src_file_path = opt.src_file_path
    trg_file_path = opt.trg_file_path
    pred_file_path = opt.pred_file_path

    if opt.export_filtered_pred:
        pred_output_file = open(os.path.join(opt.filtered_pred_path, "predictions_filtered.txt"), "w")

    score_dict = defaultdict(list)
    all_ks = process_input_ks(opt.all_ks)
    present_ks = process_input_ks(opt.present_ks)
    absent_ks = process_input_ks(opt.absent_ks)
    topk_dict = {'present': present_ks, 'absent': absent_ks, 'all': all_ks}

    total_num_src = 0
    total_num_src_with_present_keyphrases = 0
    total_num_src_with_absent_keyphrases = 0
    total_num_unique_predictions = 0
    total_num_present_filtered_predictions = 0
    total_num_present_unique_targets = 0
    total_num_absent_filtered_predictions = 0
    total_num_absent_unique_targets = 0
    total_dupratios = 0.0
    max_unique_targets = 0

    for data_idx, (src_l, trg_l, pred_l) in enumerate(
            zip(open(src_file_path), open(trg_file_path), open(pred_file_path))):
        total_num_src += 1
        # convert the str to token list
        pred_str_list = pred_l.strip().split(';')
        pred_token_2dlist = [pred_str.strip().split(' ') for pred_str in pred_str_list]

        trg_str_list = trg_l.strip().split(';')
        trg_token_2dlist = [trg_str.strip().split(' ') for trg_str in trg_str_list]

        if "<eos>" in src_l:
            [title, context] = src_l.strip().split('<eos>')
            src_token_list = title.strip().split(' ') + context.strip().split(' ')
        else:
            src_token_list = src_l.strip().split(' ')
        num_predictions = len(pred_str_list)

        # perform stemming
        stemmed_src_token_list = stem_word_list(src_token_list)

        if opt.target_already_stemmed:
            # Just for SemEval dataset
            stemmed_trg_token_2dlist = trg_token_2dlist
        else:
            stemmed_trg_token_2dlist = stem_str_list(trg_token_2dlist)

        stemmed_pred_token_2dlist = stem_str_list(pred_token_2dlist)

        # Filter out duplicate, invalid, and extra one word predictions
        filtered_stemmed_pred_token_2dlist, num_duplicated_predictions = filter_prediction(opt.disable_valid_filter,
                                                                                           opt.disable_extra_one_word_filter,
                                                                                           stemmed_pred_token_2dlist)
        total_num_unique_predictions += (num_predictions - num_duplicated_predictions)
        total_dupratios += (num_duplicated_predictions / len(stemmed_pred_token_2dlist))

        # Remove duplicated targets
        unique_stemmed_trg_token_2dlist, num_duplicated_trg = find_unique_target(stemmed_trg_token_2dlist)

        num_unique_targets = len(unique_stemmed_trg_token_2dlist)

        if num_unique_targets > max_unique_targets:
            max_unique_targets = num_unique_targets

        # separate present and absent keyphrases for predictions and targets
        present_filtered_stemmed_pred_token_2dlist, absent_filtered_stemmed_pred_token_2dlist = separate_present_absent_by_source(
            stemmed_src_token_list, filtered_stemmed_pred_token_2dlist, opt.match_by_str)
        present_unique_stemmed_trg_token_2dlist, absent_unique_stemmed_trg_token_2dlist = separate_present_absent_by_source(
            stemmed_src_token_list, unique_stemmed_trg_token_2dlist, opt.match_by_str)

        total_num_present_filtered_predictions += len(present_filtered_stemmed_pred_token_2dlist)
        total_num_present_unique_targets += len(present_unique_stemmed_trg_token_2dlist)
        total_num_absent_filtered_predictions += len(absent_filtered_stemmed_pred_token_2dlist)
        total_num_absent_unique_targets += len(absent_unique_stemmed_trg_token_2dlist)
        if len(present_unique_stemmed_trg_token_2dlist) > 0:
            total_num_src_with_present_keyphrases += 1
        if len(absent_unique_stemmed_trg_token_2dlist) > 0:
            total_num_src_with_absent_keyphrases += 1

        # compute all the metrics and update the score_dict
        score_dict = update_score_dict(unique_stemmed_trg_token_2dlist, filtered_stemmed_pred_token_2dlist,
                                       topk_dict['all'], score_dict, 'all')
        # compute all the metrics and update the score_dict for present keyphrase
        score_dict = update_score_dict(present_unique_stemmed_trg_token_2dlist,
                                       present_filtered_stemmed_pred_token_2dlist,
                                       topk_dict['present'], score_dict, 'present')
        # compute all the metrics and update the score_dict for present keyphrase
        score_dict = update_score_dict(absent_unique_stemmed_trg_token_2dlist,
                                       absent_filtered_stemmed_pred_token_2dlist,
                                       topk_dict['absent'], score_dict, 'absent')
        if opt.export_filtered_pred:
            final_pred_str_list = []
            for word_list in filtered_stemmed_pred_token_2dlist:
                final_pred_str_list.append(' '.join(word_list))
            pred_print_out = ';'.join(final_pred_str_list) + '\n'
            pred_output_file.write(pred_print_out)

    if opt.export_filtered_pred:
        pred_output_file.close()
    all_dupratio = total_dupratios / total_num_src

    total_num_unique_targets = total_num_present_unique_targets + total_num_absent_unique_targets
    total_num_filtered_predictions = total_num_present_filtered_predictions + total_num_absent_filtered_predictions
    result_txt_str = ""

    # report global statistics
    result_txt_str += "===================================Dataset Details====================================\n"
    result_txt_str += (
            'Total #samples: %d\t # samples with present keyphrases: %d\t # samples with absent keyphrases: %d\n' % (
        total_num_src, total_num_src_with_present_keyphrases, total_num_src_with_absent_keyphrases))
    result_txt_str += ('Max. unique targets per src: %d\n' % max_unique_targets)

    # report statistics and scores for all predictions and targets
    result_txt_str_all, field_list_all, result_list_all = report_stat_and_scores(total_num_filtered_predictions,
                                                                                 total_num_unique_targets,
                                                                                 total_num_src, score_dict,
                                                                                 topk_dict['all'], 'all')
    result_txt_str_present, field_list_present, result_list_present = report_stat_and_scores(
        total_num_present_filtered_predictions, total_num_present_unique_targets, total_num_src, score_dict,
        topk_dict['present'], 'present')
    result_txt_str_absent, field_list_absent, result_list_absent = report_stat_and_scores(
        total_num_absent_filtered_predictions, total_num_absent_unique_targets, total_num_src, score_dict,
        topk_dict['absent'], 'absent')
    result_txt_str += (result_txt_str_all + result_txt_str_present + result_txt_str_absent)
    field_list = field_list_all + field_list_present + field_list_absent
    result_list = result_list_all + result_list_present + result_list_absent

    # Write to files
    # topk_dict = {'present': [5, 10, 'M'], 'absent': [5, 10, 50, 'M'], 'all': [5, 10, 'M']}
    k_list = topk_dict['all'] + topk_dict['present'] + topk_dict['absent']
    result_file_suffix = '_'.join([str(k) for k in k_list])
    if opt.meng_rui_precision:
        result_file_suffix += '_meng_rui_precision'

    results_txt_file = open(os.path.join(opt.exp_path, "results_log_{}.txt".format(result_file_suffix)), "w")

    # Report MAE on lengths
    result_txt_str += "===================================MAE & DupRatio stat====================================\n"

    num_targets_present_array = np.array(score_dict['num_targets_present'])

    num_predictions_present_array = np.array(score_dict['num_predictions_present'])
    num_targets_absent_array = np.array(score_dict['num_targets_absent'])
    num_predictions_absent_array = np.array(score_dict['num_predictions_absent'])

    all_mae = mae(num_targets_present_array + num_targets_absent_array,
                  num_predictions_present_array + num_predictions_absent_array)
    present_mae = mae(num_targets_present_array, num_predictions_present_array)
    absent_mae = mae(num_targets_absent_array, num_predictions_absent_array)

    result_txt_str += "MAE on keyphrase numbers (all): {:.5}\n".format(all_mae)
    result_txt_str += "MAE on keyphrase numbers (present): {:.5}\n".format(present_mae)
    result_txt_str += "MAE on keyphrase numbers (absent): {:.5}\n".format(absent_mae)
    result_txt_str += "Dupratio on keyphrase numbers (all): {:.5}\n".format(all_dupratio)
    results_txt_file.write(result_txt_str)
    results_txt_file.close()

    results_tsv_file = open(os.path.join(opt.exp_path, "results_log_{}.tsv".format(result_file_suffix)), "w")
    results_tsv_file.write('\t'.join(field_list) + '\n')
    results_tsv_file.write('\t'.join('%.5f' % result for result in result_list) + '\n')
    results_tsv_file.close()

    # save score dict for future use
    score_dict_pickle = open(os.path.join(opt.exp_path, "score_dict_{}.pickle".format(result_file_suffix)), "wb")
    pickle.dump(score_dict, score_dict_pickle)
    score_dict_pickle.close()

    return


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


def mae(a, b):
    return (np.abs(a - b)).mean()


def report_stat_and_scores(total_num_filtered_predictions, num_unique_trgs, total_num_src, score_dict, topk_list,
                           present_tag):
    result_txt_str = "===================================%s====================================\n" % present_tag
    result_txt_str += "#unique targets: %d\t\t\t\t #unique targets per src:%.3f\n" % \
                      (num_unique_trgs, num_unique_trgs / total_num_src)
    result_txt_str += "#predictions after filtering: %d\t #predictions after filtering per src:%.3f\n" % \
                      (total_num_filtered_predictions, total_num_filtered_predictions / total_num_src)

    classification_output_str, classification_field_list, classification_result_list = report_classification_scores(
        score_dict, topk_list, present_tag)
    result_txt_str += classification_output_str
    field_list = classification_field_list
    result_list = classification_result_list

    return result_txt_str, field_list, result_list


def report_classification_scores(score_dict, topk_list, present_tag):
    output_str = ""
    result_list = []
    field_list = []
    for topk in topk_list:
        total_predictions_k = sum(score_dict['num_predictions@{}_{}'.format(topk, present_tag)])
        total_targets_k = sum(score_dict['num_targets@{}_{}'.format(topk, present_tag)])
        total_num_matches_k = sum(score_dict['num_matches@{}_{}'.format(topk, present_tag)])
        # Compute the micro averaged recall, precision and F-1 score
        micro_avg_precision_k, micro_avg_recall_k, micro_avg_f1_score_k = compute_classification_metrics(
            total_num_matches_k, total_predictions_k, total_targets_k)
        # Compute the macro averaged recall, precision and F-1 score
        macro_avg_precision_k = sum(score_dict['precision@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['precision@{}_{}'.format(topk, present_tag)])
        macro_avg_recall_k = sum(score_dict['recall@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['recall@{}_{}'.format(topk, present_tag)])
        macro_avg_f1_score_k = (2 * macro_avg_precision_k * macro_avg_recall_k) / (
                macro_avg_precision_k + macro_avg_recall_k) if (macro_avg_precision_k + macro_avg_recall_k) > 0 else 0.0
        output_str += (
            "Begin===============classification metrics {}@{}===============Begin\n".format(present_tag, topk))
        output_str += ("#target: {}, #predictions: {}, #corrects: {}\n".format(total_targets_k, total_predictions_k,
                                                                               total_num_matches_k))
        output_str += "Micro:\tP@{}={:.5}\tR@{}={:.5}\tF1@{}={:.5}\n".format(topk, micro_avg_precision_k, topk,
                                                                             micro_avg_recall_k, topk,
                                                                             micro_avg_f1_score_k)
        output_str += "Macro:\tP@{}={:.5}\tR@{}={:.5}\tF1@{}={:.5}\n".format(topk, macro_avg_precision_k, topk,
                                                                             macro_avg_recall_k, topk,
                                                                             macro_avg_f1_score_k)
        field_list += ['macro_avg_p@{}_{}'.format(topk, present_tag), 'macro_avg_r@{}_{}'.format(topk, present_tag),
                       'macro_avg_f1@{}_{}'.format(topk, present_tag)]
        result_list += [macro_avg_precision_k, macro_avg_recall_k, macro_avg_f1_score_k]
    return output_str, field_list, result_list


if __name__ == '__main__':
    # load settings for training
    parser = argparse.ArgumentParser(
        description='evaluate_prediction.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.post_predict_opts(parser)
    opt = parser.parse_args()

    if opt.exp_path == "" and opt.filtered_pred_path == "":
        pred_folder_path = os.path.split(opt.pred_file_path)[0]
        exp_folder_path = pred_folder_path.replace("pred/", "exp/")
        opt.exp_path = exp_folder_path
        opt.filtered_pred_path = pred_folder_path

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.filtered_pred_path = opt.filtered_pred_path % (opt.exp, opt.timemark)

    present_absent_segmenter = '<peos>'

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.filtered_pred_path):
        os.makedirs(opt.filtered_pred_path)

    logging = config.init_logging(log_file=opt.exp_path + '/evaluate_prediction_result.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
