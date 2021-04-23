import torch
import argparse
import config
import logging
import os
from pykp.model import Seq2SeqModel
from torch.optim import Adam
from utils.time_log import time_since
from utils.data_loader import load_data_and_vocab
import time
import torch.nn as nn
from pykp.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics
import math
import sys
from utils.report import export_train_and_valid_loss
from preprocess_opt import arrange_opt

EPS = 1e-8


def init_model(opt):
    logging.info('======================  Model Parameters  =========================')

    if opt.copy_attention:
        logging.info('Training a seq2seq model with copy mechanism')
    else:
        logging.info('Training a seq2seq model')
    model = Seq2SeqModel(opt)

    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        model.load_state_dict(torch.load(opt.pretrained_model))
    return model.to(opt.device)


def evaluate_valid_loss(data_loader, model, opt):
    model.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            # load one2many dataset
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _ = batch
            num_trgs = [len(trg_str_list) for trg_str_list in
                        trg_str_2dlist]  # a list of num of targets in each batch, with len=batch_size

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            trg = trg.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            trg_oov = trg_oov.to(opt.device)

            start_time = time.time()
            decoder_dist, attention_dist = model(src, src_lens, trg, src_oov, max_num_oov, src_mask, num_trgs)
            forward_time = time_since(start_time)
            forward_time_total += forward_time

            start_time = time.time()

            loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask)
            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time

            evaluation_loss_sum += loss.item()
            total_trg_tokens += sum(trg_lens)

    eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total, loss_compute_time=loss_compute_time_total)
    return eval_loss_stat


def train_model(model, optimizer_ml, train_data_loader, valid_data_loader, opt):

    logging.info('======================  Start Training  =========================')

    total_batch = -1
    early_stop_flag = False

    total_train_loss_statistics = LossStatistics()
    report_train_loss_statistics = LossStatistics()
    report_train_ppl = []
    report_valid_ppl = []
    report_train_loss = []
    report_valid_loss = []
    best_valid_ppl = float('inf')
    best_valid_loss = float('inf')
    num_stop_dropping = 0

    model.train()

    for epoch in range(opt.start_epoch, opt.epochs+1):
        if early_stop_flag:
            break

        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1

            # Training
            batch_loss_stat, decoder_dist = train_one_batch(batch, model, optimizer_ml, opt, batch_i)
            report_train_loss_statistics.update(batch_loss_stat)
            total_train_loss_statistics.update(batch_loss_stat)

            # Checkpoint, decay the learning rate if validation loss stop dropping, apply early stopping if stop decreasing for several epochs.
            # Save the model parameters if the validation loss improved.
            if total_batch % 4000 == 0:
                print("Epoch %d; batch: %d; total batch: %d" % (epoch, batch_i, total_batch))
                sys.stdout.flush()

            if epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.checkpoint_interval > -1 and total_batch > 1 and total_batch % opt.checkpoint_interval == 0):

                    # test the model on the validation dataset for one epoch
                    valid_loss_stat = evaluate_valid_loss(valid_data_loader, model, opt)
                    model.train()
                    current_valid_loss = valid_loss_stat.xent()
                    current_valid_ppl = valid_loss_stat.ppl()
                    print("Enter check point!")
                    sys.stdout.flush()

                    current_train_ppl = report_train_loss_statistics.ppl()
                    current_train_loss = report_train_loss_statistics.xent()

                    # debug
                    if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
                        logging.info(
                            "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (epoch, batch_i, total_batch))
                        exit()
                    # update the best valid loss and save the model parameters
                    if current_valid_loss < best_valid_loss:
                        print("Valid loss drops")
                        sys.stdout.flush()
                        best_valid_loss = current_valid_loss
                        best_valid_ppl = current_valid_ppl
                        num_stop_dropping = 0

                        check_pt_model_path = os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d' % (
                            opt.exp, epoch, batch_i, total_batch) + '.model')
                        torch.save(  # save model parameters
                            model.state_dict(),
                            open(check_pt_model_path, 'wb')
                        )
                        logging.info('Saving checkpoint to %s' % check_pt_model_path)

                    else:
                        print("Valid loss does not drop")
                        sys.stdout.flush()
                        num_stop_dropping += 1
                        # decay the learning rate by a factor
                        for i, param_group in enumerate(optimizer_ml.param_groups):
                            old_lr = float(param_group['lr'])
                            new_lr = old_lr * opt.learning_rate_decay
                            if old_lr - new_lr > EPS:
                                param_group['lr'] = new_lr

                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        'avg training ppl: %.3f; avg validation ppl: %.3f; best validation ppl: %.3f' % (
                            current_train_ppl, current_valid_ppl, best_valid_ppl))
                    logging.info(
                        'avg training loss: %.3f; avg validation loss: %.3f; best validation loss: %.3f' % (
                            current_train_loss, current_valid_loss, best_valid_loss))

                    report_train_ppl.append(current_train_ppl)
                    report_valid_ppl.append(current_valid_ppl)
                    report_train_loss.append(current_train_loss)
                    report_valid_loss.append(current_valid_loss)

                    if num_stop_dropping >= opt.early_stop_tolerance:
                        logging.info(
                            'Have not increased for %d check points, early stop training' % num_stop_dropping)
                        early_stop_flag = True
                        break
                    report_train_loss_statistics.clear()

    # export the training curve
    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    export_train_and_valid_loss(report_train_loss, report_valid_loss, report_train_ppl, report_valid_ppl,
                                opt.checkpoint_interval, train_valid_curve_path)


def train_one_batch(batch, model, optimizer, opt, batch_i):
    # load one2many data
    """
    src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
    src_lens: a list containing the length of src sequences for each batch, with len=batch
    src_mask: a FloatTensor, [batch, src_seq_len]
    src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
    trg: LongTensor [batch, trg_seq_len], each target trg[i] contains the indices of a set of concatenated keyphrases, separated by opt.word2idx[pykp.io.SEP_WORD]
         if opt.delimiter_type = 0, SEP_WORD=<sep>, if opt.delimiter_type = 1, SEP_WORD=<eok>
    trg_lens: a list containing the length of trg sequences for each batch, with len=batch
    trg_mask: a FloatTensor, [batch, trg_seq_len]
    trg_oov: same as trg_oov, but all unk words are replaced with temporary idx, e.g. 50000, 50001 etc.
    """
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _ = batch
    # a list of num of targets in each batch, with len=batch_size
    num_trgs = [len(trg_str_list) for trg_str_list in trg_str_2dlist]

    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    trg = trg.to(opt.device)
    trg_mask = trg_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    trg_oov = trg_oov.to(opt.device)

    optimizer.zero_grad()

    start_time = time.time()

    decoder_dist, attention_dist = model(src, src_lens, trg, src_oov, max_num_oov, src_mask, num_trgs=num_trgs)
    forward_time = time_since(start_time)

    start_time = time.time()

    loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask)

    loss_compute_time = time_since(start_time)

    total_trg_tokens = sum(trg_lens)

    if math.isnan(loss.item()):
        print("Batch i: %d" % batch_i)
        print("src")
        print(src)
        print(src_oov)
        print(src_str_list)
        print(src_lens)
        print(src_mask)
        print("trg")
        print(trg)
        print(trg_oov)
        print(trg_str_2dlist)
        print(trg_lens)
        print(trg_mask)
        print("oov list")
        print(oov_lists)
        print("Decoder")
        print(decoder_dist)
        print(attention_dist)
        raise ValueError("Loss is NaN")

    if opt.loss_normalization == "tokens":  # use number of target tokens to normalize the loss
        normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches':  # use batch_size to normalize the loss
        normalization = src.size(0)
    else:
        raise ValueError('The type of loss normalization is invalid.')

    assert normalization > 0, 'normalization should be a positive number'

    start_time = time.time()
    # back propagation on the normalized loss
    loss.div(normalization).backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

    optimizer.step()

    # construct a statistic object for the loss
    stat = LossStatistics(loss.item(), total_trg_tokens, n_batch=1, forward_time=forward_time, loss_compute_time=loss_compute_time, backward_time=backward_time)

    return stat, decoder_dist.detach()


def main(opt):
    try:
        start_time = time.time()
        train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)
        load_data_time = time_since(start_time)
        logging.info('Time for loading the data: %.1f' % load_data_time)
        start_time = time.time()
        model = init_model(opt)

        optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
        train_model(model, optimizer, train_data_loader, valid_data_loader, opt)

        training_time = time_since(start_time)
        logging.info('Time for training: %.1f' % training_time)
    except Exception as e:
        logging.exception("")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.vocab_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    opt = parser.parse_args()
    opt = arrange_opt(opt)

    logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
