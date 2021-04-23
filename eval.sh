python3 evaluate.py -pred_file_path [prediction_path/predictions.txt] -src_file_path data/cross_domain_sorted/word_inspec_testing_context.txt    -trg_file_path data/cross_domain_sorted/word_inspec_testing_context.txt   -exp inspec -export_filtered_pred -disable_extra_one_word_filter -invalidate_unk  -all_ks  5 10 M  -present_ks 5 10 M  -absent_ks 5 10 M
python3 evaluate.py -pred_file_path [prediction_path/predictions.txt] -src_file_path data/cross_domain_sorted/word_krapivin_testing_context.txt  -trg_file_path data/cross_domain_sorted/word_krapivin_testing_context.txt -exp krapivin -export_filtered_pred -disable_extra_one_word_filter -invalidate_unk  -all_ks  5 10 M  -present_ks 5 10 M  -absent_ks 5 10 M
python3 evaluate.py -pred_file_path [prediction_path/predictions.txt] -src_file_path data/cross_domain_sorted/word_nus_testing_context.txt       -trg_file_path data/cross_domain_sorted/word_nus_testing_context.txt      -exp nus -export_filtered_pred -disable_extra_one_word_filter -invalidate_unk  -all_ks  5 10 M  -present_ks 5 10 M  -absent_ks 5 10 M
python3 evaluate.py -pred_file_path [prediction_path/predictions.txt] -src_file_path data/cross_domain_sorted/word_semeval_testing_context.txt   -trg_file_path data/cross_domain_sorted/word_semeval_testing_context.txt  -exp semeval -export_filtered_pred -disable_extra_one_word_filter -invalidate_unk  -all_ks  5 10 M  -present_ks 5 10 M  -absent_ks 5 10 M
python3 evaluate.py -pred_file_path [prediction_path/predictions.txt] -src_file_path data/kp20k_sorted/test_src.txt   -trg_file_path data/kp20k_sorted/test_trg.txt  -exp kp20k -export_filtered_pred -disable_extra_one_word_filter -invalidate_unk  -all_ks  5 10 M  -present_ks 5 10 M  -absent_ks 5 10 M