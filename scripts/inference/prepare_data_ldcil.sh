


data_path="/home/anirudh/wav2vec2-train/data/dev/ldcil_big_folder_infer"
prep_scripts="/home/anirudh/common_scripts/prep_scripts"
valid_folder="ldcil_big_folder"

valid_path="/home/anirudh/wav2vec2-train/data/dev/${valid_folder}"


dataset_type="valid"
valid_percentage=0

python ${prep_scripts}/manifest.py ${data_path} --dest ${valid_path} --ext wav --train-name ${dataset_type} --valid-percent ${valid_percentage} --jobs -1
echo "Manifest Creation Done"

python ${prep_scripts}/labels.py --jobs 64 --tsv ${valid_path}/valid.tsv --output-dir ${valid_path} --output-name valid --txt-dir ${data_path}
echo "Word file generated"

cp /home/anirudh/wav2vec2-train/data/finetuning/dict.ltr.txt ${valid_path}
