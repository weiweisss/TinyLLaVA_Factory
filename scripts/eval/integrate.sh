#!/bin/bash
# integral.sh

# ===== 在这里定义通用变量 =====
MODEL_PATH="/media/Dataset/llava_checkpoints/llava_factory/tiny-llava-TinyLlama-1.1B-Chat-v1.0-clip-vit-large-patch14-llama-mof-base-finetune/"
MODEL_NAME="tiny-llava-TinyLlama-1.1B-Chat-v1.0-clip-vit-large-patch14-llama-mof-base-finetune"
EVAL_DIR="/media/Dataset/llava_dataset/eval"
CONV_MODE="llama"


# 获取脚本所在目录
script_dir=$(dirname "$0")

# ===== 调用同目录下的其他脚本，并传入参数 =====
echo "执行 textvqa.sh"
bash "$script_dir/textvqa.sh" "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR" "$CONV_MODE"

echo "执行 pope.sh"
bash "$script_dir/pope.sh" "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR" "$CONV_MODE"

echo "执行 mmvet.sh"
bash "$script_dir/mmvet.sh" "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR" "$CONV_MODE"

echo "执行 mmmu.sh"
bash "$script_dir/mmmu.sh" "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR" "$CONV_MODE"

echo "执行 mme.sh"
bash "$script_dir/mme.sh" "$MODEL_PATH" "$MODEL_NAME" "$EVAL_DIR" "$CONV_MODE"

echo "所有脚本执行完成"