import torch
import numpy as np
import random
import pykp
import os
import logging
import json


def arrange_opt(opt, stage="train"):
    """
    Just make the file name more clear.
    Args:
        opt:
        stage: if stage='train', make the exp and model path, else, make the predict path.

    Returns:

    """
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)



    if opt.one2many:
        opt.exp += '.one2many'

    if opt.copy_attention:
        opt.exp += '.copy'

    if hasattr(opt, 'bidirectional') and opt.bidirectional:
        opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

    if opt.delimiter_type == 0:
        opt.delimiter_word = pykp.io.SEP_WORD
    else:
        opt.delimiter_word = pykp.io.EOS_WORD
    # make the different file name according to the arg 'stage'
    if stage == "prediction":
        opt.exp = 'predict.' + opt.exp
        # fill time into the name
        if opt.pred_path.find('%s') > 0:
            opt.pred_path = opt.pred_path % (opt.exp, opt.timemark)

        if not os.path.exists(opt.pred_path):
            os.makedirs(opt.pred_path)
    else:
        # fill time into the name
        if opt.exp_path.find('%s') > 0:
            opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
            opt.model_path = opt.model_path % (opt.exp, opt.timemark)

        if not os.path.exists(opt.exp_path):
            os.makedirs(opt.exp_path)
        if not os.path.exists(opt.model_path):
            os.makedirs(opt.model_path)
        print(opt.model_path)
        # dump the setting (opt) to disk in order to reuse easily
        if opt.train_from:
            opt = torch.load(
                open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'rb')
            )
        else:
            torch.save(opt,
                       open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'wb')
                       )
            print(vars(opt))
            json.dump(vars(opt), open(os.path.join(opt.model_path, opt.exp + '.initial.json'), 'w'))
        logging.info('EXP_PATH : ' + opt.exp_path)
    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")
    return opt

