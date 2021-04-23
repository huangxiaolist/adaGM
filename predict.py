import torch
from sequence_generator import SequenceGenerator
import config
from pykp.io import KeyphraseDataset
from torch.utils.data import DataLoader
from predict_core import predict_with_beam_search
import pykp.io
import argparse
from pykp.model import Seq2SeqModel
from preprocess import read_tokenized_src_file
from utils.data_loader import load_vocab
from pykp.io import build_interactive_predict_dataset, KeyphraseDataset
from preprocess_opt import arrange_opt


def init_pretrained_model(opt):
    model = Seq2SeqModel(opt)
    model.load_state_dict(torch.load(opt.model))
    model.to(opt.device)
    model.eval()
    return model


def predict(test_data_loader, model, opt):
    if opt.delimiter_type == 0:
        delimiter_word = pykp.io.SEP_WORD
    else:
        delimiter_word = pykp.io.EOS_WORD
    generator = SequenceGenerator(model,
                                  bos_idx=opt.word2idx[pykp.io.BOS_WORD],
                                  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                  pad_idx=opt.word2idx[pykp.io.PAD_WORD],
                                  beam_size=opt.beam_size,
                                  threshold=opt.threshold,
                                  max_sequence_length=opt.max_length,
                                  copy_attn=opt.copy_attention,
                                  include_attn_dist=opt.include_attn_dist,
                                  length_penalty_factor=opt.length_penalty_factor,
                                  coverage_penalty_factor=opt.coverage_penalty_factor,
                                  length_penalty=opt.length_penalty,
                                  coverage_penalty=opt.coverage_penalty,
                                  cuda=opt.gpuid > -1,
                                  n_best=opt.n_best,
                                  ignore_when_blocking=opt.ignore_when_blocking,
                                  )

    predict_with_beam_search(generator, test_data_loader, opt, delimiter_word)


def build_test_dataset(opt):
    # load vocab
    word2idx, idx2word, vocab = load_vocab(opt)
    # load data
    # read tokenized text file and convert them to 2d list of words
    tokenized_src = read_tokenized_src_file(opt.src_file, remove_eos=opt.remove_title_eos, title_guided=False)

    test_one2many = build_interactive_predict_dataset(tokenized_src, word2idx, opt)
    # build the data loader
    test_one2many_dataset = KeyphraseDataset(test_one2many, word2idx=word2idx, idx2word=idx2word,
                                             delimiter_type=opt.delimiter_type, load_train=False,
                                             remove_src_eos=opt.remove_src_eos)
    test_loader = DataLoader(dataset=test_one2many_dataset,
                             collate_fn=test_one2many_dataset.collate_fn_one2many,
                             num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                             shuffle=False)
    return test_loader


def main(opt):
    # build test dataset
    test_loader = build_test_dataset(opt)

    # init the pretrained model
    model = init_pretrained_model(opt)

    # Print out predict path
    print("Prediction path: %s" % opt.pred_path)

    # predict the keyphrases of the src file and output it to opt.pred_path/predictions.txt
    predict(test_loader, model, opt)
    return opt.pred_path


if __name__ == '__main__':
    # load settings for training
    parser = argparse.ArgumentParser(
        description='predict.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.model_opts(parser)
    config.predict_opts(parser)
    config.vocab_opts(parser)
    opt = parser.parse_args()

    opt = arrange_opt(opt, stage="prediction")
    # if opt.n_best < 0:
    #     opt.n_best = None

    main(opt)
