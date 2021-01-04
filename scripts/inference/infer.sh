
fairseq_path='/home/harveen.chadha/_/wav2vec2/fairseq'
validation_dataset='dev_NPTEL'
data_path='/home/harveen.chadha/_/english-asr-challenge/data/dev'
checkpoint_path='/home/harveen.chadha/_/english-asr-challenge/checkpoints/checkpoint_best.pt'
result_path='/home/harveen.chadha/_/english-asr-challenge/results'
subset='valid'
w2l_decoder='viterbi'

# python ${fairseq_path}/examples/speech_recognition/infer.py ${data_path}/${validation_dataset} --task audio_pretraining \
# --nbest 1 --path ${checkpoint_path} --gen-subset $subset --results-path ${result_path}/${validation_dataset} --w2l-decoder ${w2l_decoder} \
# --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
# --post-process letter 



python ${fairseq_path}/examples/speech_recognition/infer.py ${data_path}/${validation_dataset} --task audio_pretraining \
--nbest 1 --path ${checkpoint_path} --gen-subset $subset --results-path ${result_path}/${validation_dataset} --w2l-decoder kenlm --lm-model /home/harveen.chadha/_/english-asr-challenge/lm/lm.binary \
--lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 --lexicon /home/harveen.chadha/_/english-asr-challenge/data/training/lexicon.lst \
--post-process letter --beam 128   