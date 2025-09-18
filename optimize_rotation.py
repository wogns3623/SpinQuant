# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
from logging import Logger

import datasets
import torch
import torch.distributed as dist
from torch import nn
from transformers import LlamaTokenizerFast, Trainer, default_data_collator
import transformers
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from train_utils.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant
from train_utils.optimizer import SGDG
from utils.data_utils import CustomJsonDataset
from utils.hadamard_utils import random_hadamard_matrix
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict
from utils import data_utils, eval_utils, utils


class RotateModule(nn.Module):
    def __init__(self, R_init):
        super(RotateModule, self).__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32).to(torch.device("cuda")))

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = get_local_rank()
    log: Logger = utils.get_logger(
        "spinquant", dist_rank=local_rank, output_dir=training_args.logging_dir
    )

    log.info("the rank is {}".format(local_rank))
    dist.barrier()

    if ptq_args.debug:
        ptq_args.optimize_cayley = False
        ptq_args.optimize_inverse = True
        ptq_args.optimize_sort_group = True

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )

    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model = prepare_model(ptq_args, model)
    for param in model.parameters():
        param.requires_grad = False
    R1 = random_hadamard_matrix(model.config.hidden_size, "cuda")
    model.R1 = RotateModule(R1)
    for i in range(model.config.num_hidden_layers):
        # Each head dim = 128 for Llama model
        R2 = random_hadamard_matrix(
            model.config.hidden_size // model.config.num_attention_heads, "cuda"
        )
        model.model.layers[i].self_attn.R2 = RotateModule(R2)

    if local_rank == 0:
        log.info("Model init completed for training {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False
    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )
    train_data = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )

    save_keys = ["R1.weight", "self_attn.R2"]

    if ptq_args.optimize_inverse:
        # find per layer inversion of rotation matrix R1 and R2
        save_keys += ["r_inv"]
    if ptq_args.optimize_sort_group:
        # find per layer best sorting group of rotation matrix R1 and R2
        save_keys += ["r_perm"]
    model.set_rotation_adjust(ptq_args.optimize_inverse, ptq_args.optimize_sort_group)

    if ptq_args.optimize_cayley:
        trainable_parameters = [model.R1.weight] + [
            model.model.layers[i].self_attn.R2.weight
            for i in range(model.config.num_hidden_layers)
        ]
        model.seqlen = training_args.model_max_length
        optimizer = SGDG(
            trainable_parameters, lr=training_args.learning_rate, stiefel=True
        )
        MyTrainer = Trainer
        # Use FSDP for 70B rotation training
        if training_args.fsdp != "" and training_args.fsdp != []:
            MyTrainer = FSDPTrainer

        trainer = MyTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=None,
            data_collator=default_data_collator,
            optimizers=(optimizer, None),
        )
        dist.barrier()

        trainer.train()
        if training_args.fsdp != "" and training_args.fsdp != []:
            cpu_state = pt_fsdp_state_dict(trainer.model)
        else:
            cpu_state = trainer.model.state_dict()
    else:
        dataloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )
        dataloader.input_ids = dataloader.input_ids[:, :4096]

        model.seqlen = training_args.model_max_length
        ptq_args.bsz = 1
        eval_utils.evaluator(model, dataloader, utils.DEV, ptq_args)
        cpu_state = model.state_dict()

    R_dict = {
        key.replace(".weight", ""): value
        for key, value in cpu_state.items()
        if any(matcher in key and value is not None for matcher in save_keys)
    }
    if local_rank == 0:
        os.makedirs(model_args.output_rotation_path, exist_ok=True)
        path = os.path.join(model_args.output_rotation_path, "R.bin")
        torch.save(
            R_dict,
            path,
        )
    dist.barrier()


if __name__ == "__main__":
    train()
