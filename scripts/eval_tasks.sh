CUDA_VISIBLE_DEVICES=0 ./scripts/eval.sh --model meta-llama/Llama-2-7b-hf --bits 4 4 4 \
  --output_dir ./logs/0918_adjust_both \
  --lm_eval \
  --lm_eval_batch_size 1 \
  --tasks piqa arc_easy arc_challenge boolq hellaswag winogrande \
  --master_port 29501 \
