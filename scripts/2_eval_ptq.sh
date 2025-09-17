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
kv_bits=$4
save_path=$5

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
--input_model $input_model \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits $w_bits \
--a_bits $a_bits \
--k_bits $kv_bits \
--v_bits $kv_bits \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--rotate \
--optimized_rotation_path "$save_path/R.bin" \

