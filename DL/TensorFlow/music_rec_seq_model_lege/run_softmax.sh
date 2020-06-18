#########################################################################
# File Name: run.sh
# Author: 
# Mail: @.com
# Created Time: Tue 17 Dec 2019 12:02:06 PM CST
#########################################################################
#!/bin/bash

export LD_LIBRARY_PATH="./cudnn/cuda/lib64:$LD_LIBRARY_PATH"

python run_dnn_softmax.py --train_dir=data --model_dir=model --pretrain_embedding=data/pretrain_embedding/pre_embedding --model_type=${model_type} --batch_size=$batch_size --shuffle_buffer_size=$shuffle_buffer_size --num_epochs=$num_epochs --rate_category_size=$rate_category_size --song_size=$song_size --singer_size=$singer_size --sex_size=$sex_size --neg_rate=$neg_rate 

echo "push to hdfs"
/data/offline/yard-hadoop/bin/hadoop fs -Dhadoop.job.ugi=tdw_letian:tianle89 -put model hdfs://ss-wxg-3-v2/user/tdw_letian/work/recommend/seq_model/model_res/${model_type}_model_`date +%Y-%m-%d_%H-%M-%S`
