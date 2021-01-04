#!/bin/bash

config_path='/home/harveen.chadha/_/experiments/experiment_2/english-asr-challenge/config'
config_name='finetuning'
data_path='/home/harveen.chadha/_/experiments/experiment_2/english-asr-challenge/data/training'
pretrained_model_path='/home/harveen.chadha/_/experiments/experiment_2/english-asr-challenge/checkpoints/checkpoint_best.pt'
PORT=-1
checkpoints_path='/home/harveen.chadha/_/experiments/experiment_2/english-asr-challenge/checkpoints/finetuning'
log_path='/home/harveen.chadha/_/experiments/experiment_2/english-asr-challenge/logs/finetuning'
tensorboard_path=${log_path}/tensorboard
gpus=8
run_in_nohup=0  #0 for no, 1 for yes




update_freq=$((24/${gpus}))
#update_freq=''$update_freq
echo ${update_freq}

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}




if [ "${run_in_nohup}" = 1 ]; then

	local_timestamp=$(timestamp)
	tensorboard_path=${tensorboard_path}_${local_timestamp}
	mkdir -p ${tensorboard_path}
	echo ${local_timestamp}
	echo ${tensorboard_path}

	nohup fairseq-hydra-train \
	    distributed_training.distributed_port=${PORT} \
	        task.data=${data_path} \
		    model.w2v_path=${pretrained_model_path} \
		    distributed_training.distributed_world_size=${gpus} \
		    +optimization.update_freq=[$update_freq] \
			+common.tensorboard_logdir=${tensorboard_path} \
			checkpoint.save_dir=${checkpoints_path} \
			checkpoint.restore_file=${checkpoints_path}/checkpoint_last.pt \
		        --config-dir ${config_path} \
			    --config-name ${config_name} &> ${log_path}/${local_timestamp}.out & 

	nohup tensorboard --logdir ${tensorboard_path} --bind_all &> /dev/null &

else

	fairseq-hydra-train \
	    distributed_training.distributed_port=${PORT} \
	        task.data=${data_path} \
		    model.w2v_path=${pretrained_model_path} \
		    distributed_training.distributed_world_size=${gpus} \
		    +optimization.update_freq=[$update_freq] \
			checkpoint.restore_file=${checkpoints_path}/checkpoint_last.pt \
		        --config-dir ${config_path} \
			    --config-name ${config_name}

fi