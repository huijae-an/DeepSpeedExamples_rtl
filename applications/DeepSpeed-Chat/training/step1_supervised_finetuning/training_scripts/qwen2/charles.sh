#!/bin/bash

OUTPUT=./qwen2.5-7b-coder-instruct-pymtl_gpt_4o_augmented
GARBAGE=/scratch/charleshong/garbage/*
PREVIOUS_MODEL_CACHE=$HOME/.cache/huggingface/hub/*
ZERO_STAGE=3
mkdir -p $OUTPUT
# rm -rf $GARBAGE
# rm -rf $PREVIOUS_MODEL_CACHE


deepspeed --master_port 50000 main.py \
   --data_path local/jsonfile \
   --data_split "10,0,0" \
   --data_output_path /scratch/charleshong/garbage \
   --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 1024 \
   --learning_rate 1e-5 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --dtype bf16 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --lora_dim 32 \
   --lora_module_name "layers." \
   --output_dir $OUTPUT \
   | tee $OUTPUT/training.log
