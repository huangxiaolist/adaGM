"""
statistic_correct_first_words_in_predictions.
Equals our Table 7
"""
from collections import Counter
from nltk.stem.porter import *
stemmer = PorterStemmer()


def stem_word_list(word_list):
    return [stemmer.stem(w.strip().lower()) for w in word_list]


def statistic_correct_first_words_in_predictions(src_path, pred_path):
    num_truth = 0
    num_pred = 0
    num_recall = 0
    for idx, (src_line, pred_line) in enumerate(zip(open(src_path), open(pred_path))):

        src_trgs = src_line.strip()
        pred_trgs = pred_line.strip()
        list_src_first = []
        list_pred_first = []
        # process the kp-rl predictions
        if "sep" in src_path:
            src_trgs = src_trgs.replace("<peos>", "")
        if "sep" in pred_path:
            pred_trgs = pred_trgs.replace("<peos>", "")

        for w in src_trgs.split(";"):
            w = w.split()
            if w:
                stemmed_src_token_list = stem_word_list(w)
                list_src_first.append(stemmed_src_token_list[0])

        num_truth += len(list_src_first)
        dict_src = dict(Counter(list_src_first))

        for w in pred_trgs.split(";"):
            if w:
                w = w.split()
                stemmed_pred_token_list = stem_word_list(w)
                list_pred_first.append(stemmed_pred_token_list[0])
        dict_pred = dict(Counter(list_pred_first))

        for p in list_pred_first:
            if p in dict_src and dict_src[p] > 0:
                dict_src[p] -= 1

        num_pred += sum(dict_pred.values())
        num_recall += sum(dict_src.values())

    print(num_pred, num_truth-num_recall)


if __name__ == "__main__":
    src_pred_paths = [["src_trg_file", "pred_trg_file"]]
    for src_pred_path in src_pred_paths:
        src_path, pred_path = src_pred_path
        statistic_correct_first_words_in_predictions(src_path, pred_path)


