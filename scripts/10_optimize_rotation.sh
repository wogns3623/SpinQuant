# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
input_model=$1
w_bits=$2
a_bits=$3
k_bits=$4
v_bits=$4
save_path=$5

torchrun --nnodes=1 --nproc_per_node=1 optimize_rotation.py \
--input_model $input_model  \
--output_rotation_path $save_path \
--output_dir $save_path \
--logging_dir $save_path \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 100 \
--w_bits $w_bits \
--a_bits $a_bits \
--k_bits $k_bits \
--v_bits $k_bits \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
# --optimize_rotation \
