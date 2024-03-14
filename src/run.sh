#!/bin/sh

export PATH=$PWD:$PATH
export PYTHONPATH=$PWD:$PYTHONPATH

gpu_id='0,1'
continue_from=
if [ -z ${continue_from} ]; then
	log_name='MoMuSE'$(date '+%Y-%m-%d(%H:%M:%S)')
	mkdir -p logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=3096 \
main.py \
\
--log_name $log_name \
\
--visual_direc '/mntcephfs/lee_dataset/separation/voxceleb2/mp4/' \
--mix_lst_path '../data_preparation/mixture_data_list_2mix_with_occludded.csv' \
--mixture_direc '/mntcephfs/lee_dataset/separation/voxceleb2/mixture/' \
--C 2 \
--epochs 100 \
--max_length 6 \
--accu_grad 0 \
--batch_size 6 \
--num_workers 6 \
--use_tensorboard 1 \
--lr 1e-4 \
>logs/$log_name/console2.log 
# --continue_from ${continue_from} \












