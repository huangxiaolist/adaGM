This repository contains the code for AAAI-21 paper "[Adaptive Beam Search Decoding for Discrete Keyphrase Generation](#)".

Our implementation is built on the:
- [seq2seq-keyphrase-pytorch](https://github.com/memray/seq2seq-keyphrase-pytorch).
- [a related repository](https://github.com/atulkum/pointer_summarizer). 
- beam search code: [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- [keyphrase-generation-rl](https://github.com/kenchan0226/keyphrase-generation-rl)

## Dependencies
* python 3.5+
* pytorch 1.6.0

## Dataset
We adopt the same five datasets with [keyphrase-generation-rl](https://github.com/kenchan0226/keyphrase-generation-rl), which can be downloaded from their [repository](https://github.com/kenchan0226/keyphrase-generation-rl#dataset).  
We are high acknowledgments to Mr. Wang Chen and Hou Pong Chan for their help on data preprocessing (sort present keyphrases & remove duplicated docs).

## Preprocessing
Command: `python3 preprocess.py -data_dir data/kp20k_sorted -remove_eos`

## Train
Command: `python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -one2many -batch_size 12 -seed 9527 -delimiter_type 1`

## Predict
Command: `python3 predict.py -vocab data/kp20k_sorted/ -src_file [src_file_path] -pred_path pred/%s.%s   -copy_attention -one2many  -delimiter_type 1  -model [model_path]  -max_length 60 -remove_title_eos -n_best -1 -max_eos_per_output_seq 1 -replace_unk -beam_size 20 -batch_size 1`

## Evaluate
Command: `python3 evaluate_prediction.py -pred_file_path [path_to_predictions.txt] -src_file_path [path_to_test_set_src_file] -trg_file_path [path_to_test_set_trg_file] -exp kp20k -export_filtered_pred -disable_extra_one_word_filter -invalidate_unk -all_ks 5 M -present_ks 5 M -absent_ks 5 M`

## Baselines predictions
We train the five keyphrase generation models ([CopyRNN](https://www.aclweb.org/anthology/P17-1054.pdf), [CorrRNN](https://www.aclweb.org/anthology/D18-1439.pdf), [TG-Net](https://ojs.aaai.org//index.php/AAAI/article/view/4587), [catSeq](https://www.microsoft.com/en-us/research/publication/generating-diverse-numbers-of-diverse-keyphrases/), [catSeqD](https://www.microsoft.com/en-us/research/publication/generating-diverse-numbers-of-diverse-keyphrases/))) and save the best [ckpt](https://drive.google.com/file/d/1kEL53UDzYkNkWg4DGIIchVwmNiHTAJZr/view?usp=sharing) and [predictions](https://drive.google.com/file/d/1EZ0WfPyFtFsr56FgrYugmTdTJdYgd0zm/view?usp=sharing), separately.
We also collect two models' predictions, which are given by the authors: [ExHiRD](https://www.aclweb.org/anthology/2020.acl-main.103.pdf) & [Kp-RL](https://www.aclweb.org/anthology/P19-1208.pdf).
They can be downloaded from [here](https://drive.google.com/file/d/1EZ0WfPyFtFsr56FgrYugmTdTJdYgd0zm/view?usp=sharing).

## More Details
### Reset State Mechanism Part
The code:
### Filter Mechanims
The code:

