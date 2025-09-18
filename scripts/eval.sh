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


torchrun --nnodes=1 --nproc_per_node=1 --master_port=$master_port ptq.py \
  --input_model $input_model \
  --optimized_rotation_path "$save_path/R.bin" \
  --logging_dir $output_dir \
  --do_train False \
  --do_eval True \
  --per_device_eval_batch_size 4 \
  --model_max_length 2048 \
  --fp16 False \
  --bf16 True \
  --save_safetensors False \
  --w_bits $w_bits --a_bits $a_bits --k_bits $kv_bits --v_bits $kv_bits \
  --w_clip --a_asym --k_asym --v_asym \
  --k_groupsize 128 --v_groupsize 128 \
  --rotate \
  $ARGS

