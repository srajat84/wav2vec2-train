
### Values to change -start ###

prep_scripts="/home/anirudh/common_scripts/prep_scripts"
data_path="/home/anirudh/2020_kn_1/speaker_split_ldcil/test_files"
valid_percentage=0

### Values to change end ###


dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

destination_path=$parentdir'/data/finetuning'


python ${prep_scripts}/manifest.py ${data_path} --dest ${destination_path} --ext wav --train-name valid --valid-percent ${valid_percentage} --jobs -1
echo "Manifest Creation Done"

#python ${prep_scripts}/labels.py --jobs 64 --tsv ${destination_path}/train.tsv --output-dir ${destination_path} --output-name train --txt-dir ${data_path}
python ${prep_scripts}/labels.py --jobs 64 --tsv ${destination_path}/valid.tsv --output-dir ${destination_path} --output-name valid --txt-dir ${data_path}

echo "Word file generated"

#python ${prep_scripts}/dict_and_lexicon_maker.py --wrd ${destination_path}/train.wrd --lexicon ${destination_path}/lexicon.lst --dict ${destination_path}/dict.ltr.txt
echo "Dict file generated"
