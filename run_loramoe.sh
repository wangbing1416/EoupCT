export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# export NCCL_NET=IB
# export NCCL_IB_HCA=mlx5_0
# export NCCL_DEBUG=info

lr=0.0002
lora_rank=6
lora_alpha=32
lora_trainable="gate_proj,down_proj,up_proj"
lora_dropout=0.05
lora_nums=6
blc_alpha=0.0
blc_weight=0.0


: "${MODEL_DIR:?Set MODEL_DIR to your local model checkpoint directory before running this script}"

pretrained_model=$MODEL_DIR
tokenizer_path=$MODEL_DIR
dataset_dir=./data/SuperNI
validation_file=./data/superNI

per_device_train_batch_size=2
per_device_eval_batch_size=8
gradient_accumulation_steps=4
max_seq_length=1024
seed=2
output_dir=./output
exp_name=llama-3b-lora-seed${seed}-0918


deepspeed_config_file=ds_zero2_no_offload.json
#deepspeed_config_file=config/stage2_without_offload.config
#deepspeed_config_file=ds_zero3_offload.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CUDA_LAUNCH_BLOCKING=1 \
#torchrun --nnodes 1 --nproc_per_node 8 --node_rank 0 --master_port 29502 run_loramoe.py \
deepspeed --num_nodes 1 --num_gpus 8 run_loramoe.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --dataloader_num_workers 4 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --bf16 \
    --seed ${seed} \
    --num_train_epochs 10 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --logging_dir ./tensorboard_log \
    --save_strategy steps \
    --save_total_limit 5 \
    --eval_strategy no \
    --eval_steps 5000 \
    --save_steps 5000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --max_target_length 50 \
    --output_dir ${output_dir}/${exp_name} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_nums ${lora_nums} \
    --blc_alpha ${blc_alpha} \
    --blc_weight ${blc_weight} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --ddp_find_unused_parameters False \
    --flash_attn \
    --attn_implementation flash_attention_2 \
    --overwrite_output_dir \
    --report_to tensorboard \
#    &> ./output/log/${exp_name}.log
