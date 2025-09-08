#!/usr/bin/env bash

python finetune_t3.py \
--output_dir ./checkpoints/chatterbox_finetuned \
--model_name_or_path ResembleAI/chatterbox \
--metadata_file output/metadata.csv \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--learning_rate 5e-5 \
--warmup_steps 100 \
--logging_steps 10 \
--eval_strategy steps \
--eval_steps 2000 \
--save_strategy steps \
--save_steps 4000 \
--save_total_limit 4 \
--fp16 True \
--report_to tensorboard \
--text_column_name transcript \
--audio_column_name filepath \
--eval_split_size 0.0002 \
--do_train \
--do_eval