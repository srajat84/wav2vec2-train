



fairseq_path='/opt/fairseq'
validation_dataset='ldcil_big_folder'

dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"


data_path=${parentdir}'/data/dev/'${validation_dataset}
checkpoint_path=${parentdir}'/checkpoints/finetuning/checkpoint_best.pt'
result_path=${parentdir}'/results/'${validation_dataset}
subset='valid'
w2l_decoder='viterbi'
lm=0 # 0 for viterbi and 1 for kenlm
beam=128

if [ "${lm}" = 0 ]; then


	python ${fairseq_path}/examples/speech_recognition/infer.py ${data_path} --task audio_pretraining \
	--nbest 1 --path ${checkpoint_path} --gen-subset $subset --results-path ${result_path} --w2l-decoder ${w2l_decoder} \
	--lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 6000000 \
	 --post-process letter 
else


	python ${fairseq_path}/examples/speech_recognition/infer.py ${data_path} --task audio_pretraining \
	--nbest 1 --path ${checkpoint_path} --gen-subset $subset --results-path ${result_path}_kenlm --w2l-decoder ${w2l_decoder} --lm-model ${parentdir}/data/lm/lm.binary \
	--lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 6000000 --lexicon ${parentdir}/data/finetuning/lexicon.lst \
	--post-process letter --beam ${beam}

fi	
