OUTPUT=./output_codellama_7b_lora_bf16
ZERO_STAGE=3
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path local/jsonfile \
   --data_split "10,0,0" \
   --data_output_path /scratch/huijaean/garbage \
   --model_name_or_path meta-llama/CodeLlama-7b-Instruct-hf \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 32 \
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
