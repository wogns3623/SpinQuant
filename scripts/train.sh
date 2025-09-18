output_dir=""
input_model=""
w_bits=4
a_bits=4
kv_bits=4
master_port=29500

# Parse model and output_dir arguments only
ARGS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      input_model="$2"
      shift 2
      ;;
    --output_dir)
      output_dir="$2"
      shift 2
      ;;
    --bits)
      w_bits="$2"
      a_bits="$3"
      kv_bits="$4"
      shift 4
      ;;
    --master_port)
      master_port="$2"
      shift 2
      ;;
    *)
      ARGS="$ARGS $1"
      shift
      ;;
  esac
done

torchrun --nnodes=1 --nproc_per_node=1 --master_port=$master_port optimize_rotation.py \
  --input_model $input_model  \
  --output_rotation_path $output_dir \
  --output_dir $output_dir \
  --logging_dir $output_dir \
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
  --w_bits $w_bits --a_bits $a_bits --k_bits $kv_bits --v_bits $kv_bits \
  --w_clip --a_asym --k_asym --v_asym \
  --k_groupsize 128 --v_groupsize 128 \
  $ARGS
