#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
## Always setting zero_stage to 3.
# if [ "$ZERO_STAGE" == "" ]; then
#    ZERO_STAGE=0
# fi
##
mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py --model_name_or_path facebook/opt-1.3b \
   --gradient_accumulation_steps 1 --lora_dim 128 --zero_stage 3 \
   --per_device_train_batch_size 1 \
   --gradient_checkpointing \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
