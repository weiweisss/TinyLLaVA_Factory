#!/bin/bash

#================================================================
# LLaVA 数据集后台下载脚本
#================================================================
# 功能:
# 1. 在指定的路径 /media/Dataset 下下载和整理所有数据集。
# 2. 自动检查是否已下载和解压，避免重复操作。
# 3. 自动解压并按照指定结构进行整理。
# 4. 支持后台运行和日志记录。
#
# 使用方法:
# 1. 确保路径 /media/Dataset 已存在。
# 2. 确保已安装 wget, unzip。
# 3. chmod +x download_llava_dataset.sh
# 4. nohup ./download_llava_dataset.sh > download.log 2>&1 &
# 5. tail -f download.log
# 6. pkill -f download_llava_dataset.sh
#================================================================

# 当任何命令失败时立即退出脚本
set -e

# --- 1. 设置并进入目标目录 ---
# 您指定的根目录
BASE_DIR="/media/Dataset"

# 检查根目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 目标目录 $BASE_DIR 不存在。请先创建该目录。"
    exit 1
fi

# 数据将存放在此目录下
DATA_DIR="$BASE_DIR/llava_dataset"

echo "=================================================="
echo "目标工作目录: $DATA_DIR"
echo "开始创建目录结构并进入..."
echo "=================================================="

# 创建存放数据的目录结构
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# 创建各个子目录
mkdir -p llava/llava_pretrain/images coco/train2017 gqa/images ocr_vqa/images textvqa/train_images vg/VG_100K vg/VG_100K_2 text_files

echo "目录结构创建完毕，当前工作目录: $(pwd)"
echo ""

# --- 辅助函数：检查目录是否已存在 ---
check_and_skip() {
    local dir_path=$1
    local step_name=$2

    if [ -d "$dir_path" ] && [ "$(ls -A $dir_path)" ]; then
        echo "检测到 $step_name 已存在且非空，跳过下载和解压步骤。"
        echo ""
        return 0
    else
        return 1
    fi
}

# --- 2. 下载 LLaVA Pretrain 数据集 ---
echo "=================================================="
echo "[1/8] 检查 LLaVA Pretrain 数据集..."
echo "=================================================="
if ! check_and_skip "$DATA_DIR/llava/llava_pretrain/images" "LLaVA Pretrain 数据集"; then
    echo "开始下载 LLaVA Pretrain 数据集..."

    # 创建临时目录用于下载
    mkdir -p "$DATA_DIR/llava/llava_pretrain/temp"
    cd "$DATA_DIR/llava/llava_pretrain/temp"

    # 使用 wget 从镜像站下载
    echo "下载 images.zip..."
    wget -c "https://hf-mirror.com/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip?download=true" -O images.zip

    echo "下载 blip_laion_cc_sbu_558k.json..."
    wget -c "https://hf-mirror.com/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json?download=true" -O blip_laion_cc_sbu_558k.json

    echo "下载 blip_laion_cc_sbu_558k_meta.json..."
    wget -c "https://hf-mirror.com/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k_meta.json?download=true" -O blip_laion_cc_sbu_558k_meta.json

    echo "解压 images.zip..."
    unzip images.zip -d ../images/
    rm images.zip

    # 移动 JSON 文件到 text_files
    mv blip_laion_cc_sbu_558k.json blip_laion_cc_sbu_558k_meta.json "$DATA_DIR/text_files/"

    # 清理临时目录
    cd "$DATA_DIR"
    rm -rf "$DATA_DIR/llava/llava_pretrain/temp"

    echo "LLaVA Pretrain 数据集下载和解压完成。"
    echo ""
fi

# --- 3. 下载 LLaVA SFT 文本数据 ---
echo "=================================================="
echo "[2/8] 检查 LLaVA SFT 文本数据..."
echo "=================================================="
if [ ! -f "$DATA_DIR/text_files/llava_v1_5_mix665k.json" ]; then
    echo "开始下载 LLaVA SFT 文本数据..."
    wget -O "$DATA_DIR/text_files/llava_v1_5_mix665k.json" "https://hf-mirror.com/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json?download=true"
    echo "LLaVA SFT 文本数据下载完成。"
    echo ""
else
    echo "检测到 llava_v1_5_mix665k.json 已存在，跳过下载步骤。"
    echo ""
fi

# --- 4. 下载 COCO 图片数据 ---
echo "=================================================="
echo "[3/8] 检查 COCO train2017 图片集..."
echo "=================================================="
if ! check_and_skip "$DATA_DIR/coco/train2017" "COCO train2017 数据集"; then
    echo "开始下载 COCO train2017 图片集..."
    cd "$DATA_DIR/coco"
    wget -c http://images.cocodataset.org/zips/train2017.zip
    echo "解压 COCO train2017..."
    unzip train2017.zip -d .
    rm train2017.zip
    echo "COCO 数据集处理完成。"
    echo ""
fi

# --- 5. 下载 GQA 图片数据 ---
echo "=================================================="
echo "[4/8] 检查 GQA 图片集..."
echo "=================================================="
if ! check_and_skip "$DATA_DIR/gqa/images" "GQA 数据集"; then
    echo "开始下载 GQA 图片集..."
    cd "$DATA_DIR/gqa"
    wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
    echo "解压 GQA..."
    unzip images.zip -d .
    rm images.zip
    echo "GQA 数据集处理完成。"
    echo ""
fi

# --- 6. 下载 TextVQA 图片数据 ---
echo "=================================================="
echo "[5/8] 检查 TextVQA 图片集..."
echo "=================================================="
if ! check_and_skip "$DATA_DIR/textvqa/train_images" "TextVQA 数据集"; then
    echo "开始下载 TextVQA 图片集..."
    cd "$DATA_DIR/textvqa"
    wget -c https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
    echo "解压 TextVQA..."
    unzip train_val_images.zip
    mv train_images train_images_temp
    mv train_images_temp "$DATA_DIR/textvqa/train_images"
    rm train_val_images.zip
    echo "TextVQA 数据集处理完成。"
    echo ""
fi

# --- 7. 下载 VisualGenome 图片数据 (Part 1) ---
echo "=================================================="
echo "[6/8] 检查 VisualGenome 图片集 (Part 1)..."
echo "=================================================="
if ! check_and_skip "$DATA_DIR/vg/VG_100K" "VisualGenome Part 1"; then
    echo "开始下载 VisualGenome 图片集 (Part 1)..."
    cd "$DATA_DIR/vg"
    wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O VG_100K.zip
    echo "解压 VisualGenome Part 1..."
    unzip VG_100K.zip -d .
    rm VG_100K.zip
    echo "VisualGenome Part 1 处理完成。"
    echo ""
fi

# --- 8. 下载 VisualGenome 图片数据 (Part 2) ---
echo "=================================================="
echo "[7/8] 检查 VisualGenome 图片集 (Part 2)..."
echo "=================================================="
if ! check_and_skip "$DATA_DIR/vg/VG_100K_2" "VisualGenome Part 2"; then
    echo "开始下载 VisualGenome 图片集 (Part 2)..."
    cd "$DATA_DIR/vg"
    wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O VG_100K_2.zip
    echo "解压 VisualGenome Part 2..."
    unzip VG_100K_2.zip -d .
    rm VG_100K_2.zip
    echo "VisualGenome Part 2 处理完成。"
    echo ""
fi

# --- 9. 下载 OCR-VQA 图片数据 ---
echo "=================================================="
echo "[8/8] 检查 OCR-VQA 图片集..."
echo "=================================================="
if ! check_and_skip "$DATA_DIR/ocr_vqa/images" "OCR-VQA 数据集"; then
    echo "开始下载 OCR-VQA 图片集"
    cd "$DATA_DIR/ocr_vqa"
    wget -c "https://hf-mirror.com/datasets/weizhiwang/llava_v15_instruction_images/resolve/main/ocr_vqa_images_llava_v15.zip?download=true" -O ocr_vqa_images_llava_v15.zip
    echo "解压 OCR-VQA 图片集..."
    unzip ocr_vqa_images_llava_v15.zip -d .
    rm ocr_vqa_images_llava_v15.zip
    echo "OCR-VQA 数据集处理完成。"
    echo ""
fi

# --- 完成 ---
echo "=================================================="
echo "所有数据集下载和整理工作已全部完成！"
echo "数据已全部存放在 $DATA_DIR"
echo "=================================================="