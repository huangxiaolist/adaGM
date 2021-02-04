#copyrnn
echo '======================================COPYRNN============================================='
python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -model_path model/%s.%s -exp kp20k_copyrnn -epochs 20 -copy_attention -train_ml -gpuid 0  -batch_size 12 -seed 9527
#catseq
echo '======================================CATSEQ============================================='
python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -model_path model/%s.%s -exp kp20k_catseq -epochs 20 -copy_attention -train_ml -one2many -one2many_mode 1 -gpuid 0   -batch_size 12 -seed 9527
#catseqd
echo '======================================CATSEQD============================================='
python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -model_path model/%s.%s -exp kp20k_catseqd -epochs 20 -copy_attention -orthogonal_loss -lambda_orthogonal 0.03 -train_ml -one2many -one2many_mode 1 -gpuid 0   -use_target_encoder -batch_size 12 -seed 9527
#corrRNN
echo '======================================CORRRNN============================================='
python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -model_path model/%s.%s -exp kp20k_corrrnn -epochs 20 -copy_attention -coverage_attn -review_attn -train_ml -one2many -one2many_mode 2 -delimiter_type 1 -gpuid 0   -batch_size 12 -seed 9527
#TG-NET
echo '======================================TG-NET============================================='
python3 train.py -data data/kp20k_tg_sorted/ -vocab data/kp20k_tg_sorted/ -exp_path exp/%s.%s -model_path model/%s.%s -exp kp20k_tg_net -epochs 20 -copy_attention -title_guided -train_ml -gpuid 0   -batch_size 12 -seed 9527
#AdaGM
echo '======================================AdaGM============================================='
python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -model_path model/%s.%s -exp kp20k_corrrnn -epochs 20 -copy_attention -train_ml -one2many -one2many_mode 2 -delimiter_type 1 -gpuid 0   -batch_size 12 -seed 9527
