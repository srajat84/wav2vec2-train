valid_path="/home/harveen.chadha/_/english-asr-challenge/data/dev"
prep_scripts="/home/harveen.chadha/common_scripts/prep_scripts"
valid_folder="dev_NPTEL"
data_path="/home/jupyter/english_asr_challenge/${valid_folder}"
dataset_type="valid"
valid_percentage=0

python ${prep_scripts}/manifest.py ${data_path} --dest ${valid_path}/${valid_folder} --ext wav --train-name ${dataset_type} --valid-percent ${valid_percentage} --jobs -1
echo "Manifest Creation Done"

python ${prep_scripts}/labels.py --jobs 64 --tsv ${valid_path}/${valid_folder}/valid.tsv --output-dir ${valid_path}/${valid_folder} --output-name valid --txt-dir ${data_path}
echo "Word file generated"