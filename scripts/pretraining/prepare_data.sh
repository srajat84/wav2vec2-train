train_path="/home/harveen.chadha/_/experiments/experiment_2/english-asr-challenge/data/pretraining"
prep_scripts="/home/harveen.chadha/common_scripts/prep_scripts"
data_path="/home/jupyter/english_asr_challenge/train_NPTEL_IITM/"
dataset_type="train"
valid_percentage=0.03

python ${prep_scripts}/manifest.py ${data_path} --dest ${train_path} --ext wav --train-name train --valid-percent ${valid_percentage} --jobs -1
echo "Manifest Creation Done"

# python ${prep_scripts}/labels.py --jobs 64 --tsv ${train_path}/train.tsv --output-dir ${train_path} --output-name train --txt-dir ${data_path}
# python ${prep_scripts}/labels.py --jobs 64 --tsv ${train_path}/valid.tsv --output-dir ${train_path} --output-name valid --txt-dir ${data_path}

# echo "Word file generated"

# python ${prep_scripts}/dict_and_lexicon_maker.py --wrd ${train_path}/train.wrd --lexicon ${train_path}/lexicon.lst --dict ${train_path}/dict.ltr.txt
# echo "Dict file generated"
