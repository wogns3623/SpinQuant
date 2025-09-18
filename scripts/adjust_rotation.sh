 CUDA_VISIBLE_DEVICES=1 ./train.sh --model meta-llama/Llama-2-7b-hf --bits 4 4 4 \
  --output_dir ./logs/0918_adjust_both \
  --master_port 29501 \
  --optimize_inverse --optimize_sort_group \
