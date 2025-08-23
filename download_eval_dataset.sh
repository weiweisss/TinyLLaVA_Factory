#!/bin/bash

# 本脚本将在 /media/Dataset/llava_dataset 目录下创建 eval 文件夹，
# 并下载所有评估所需的数据集。
#
# 运行此脚本前，请确保您已安装 wget, unzip, 和 gdown。
# gdown 可以通过以下命令安装: pip install gdown

# 如果任何命令执行失败，则立即退出脚本
set -e

# --- 设置目标目录 ---
TARGET_BASE_DIR="/media/Dataset/llava_dataset"

# 1. 检查并创建目标基础目录，然后进入该目录
echo "==> 指定的基础目录是: $TARGET_BASE_DIR"
mkdir -p "$TARGET_BASE_DIR"
cd "$TARGET_BASE_DIR"
echo "==> 当前工作目录已切换至: $(pwd)"
echo "============================================="

# 2. 创建主 eval 目录
echo "==> 正在创建 eval 目录..."
mkdir -p eval

# 3. 下载并解压基础评估文件 (包含所有JSON文件和目录结构)
# 用户提示：此步骤将下载包含标注、脚本和文件夹结构的基础文件。
echo "==> 正在下载并解压基础评估文件 (eval.zip)..."
#gdown '1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy' -O eval.zip
#unzip -q eval.zip -d eval/
#rm eval.zip
echo "==> 基础文件解压完成。"
echo "============================================="

# 4. 为每个基准测试下载所需的图像数据

# VQAv2
echo "==> 正在下载 VQAv2 数据集 (test2015 图像)..."
#wget -q --show-progress http://images.cocodataset.org/zips/test2015.zip -P eval/vqav2/
#unzip -q eval/vqav2/test2015.zip -d eval/vqav2/
#rm eval/vqav2/test2015.zip
echo "==> VQAv2 下载完成。"
echo "============================================="

# GQA
# 注意: 这是一个非常大的文件 (22GB)，下载过程可能需要很长时间。
echo "==> 正在下载 GQA 数据集 (图像)..."
#wget -q --show-progress https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -P eval/gqa/
#unzip -q eval/gqa/images.zip -d eval/gqa/
#rm eval/gqa/images.zip
echo "==> GQA 下载完成。"
echo "============================================="

# ScienceQA
echo "==> 正在下载 ScienceQA 数据集 (图像和JSON文件)..."
#gdown --id 1o-n4S20t3E3nS-rNl-IUXg2D-Ahb2N0e -O eval/scienceqa/data.zip
# 解压到临时目录以保持主文件夹整洁
#mkdir -p eval/scienceqa/temp_sqa
#unzip -q eval/scienceqa/data.zip -d eval/scienceqa/temp_sqa
# 压缩包包含一个 'data' 文件夹, 将其内容移动到正确位置
#mv eval/scienceqa/temp_sqa/data/* eval/scienceqa/
#rm -r eval/scienceqa/temp_sqa
#rm eval/scienceqa/data.zip
echo "==> ScienceQA 下载完成。"
echo "============================================="

# TextVQA 一键安装
echo "==> 正在下载 TextVQA 数据集 (图像)..."
#wget -q --show-progress https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -P eval/textvqa/
#unzip -q eval/textvqa/train_val_images.zip -d eval/textvqa/
#wget -q --show-progress https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json -P eval/textvqa/
#rm eval/textvqa/train_val_images.zip
echo "==> TextVQA 下载完成。"
echo "============================================="

# POPE 一键安装
echo "==> 正在下载 POPE 数据集 (COCO val2014 图像)..."
#wget -q --show-progress http://images.cocodataset.org/zips/val2014.zip -P eval/pope/
#unzip -q eval/pope/val2014.zip -d eval/pope/
#rm eval/pope/val2014.zip
#wget -q --show-progress https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_adversarial.json -P eval/pope/coco/
#wget -q --show-progress https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_popular.json -P eval/pope/coco/
#wget -q --show-progress https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json -P eval/pope/coco/
echo "==> POPE 下载完成。"
echo "============================================="

# MME 一键安装
echo "==> 正在下载 MME 数据集..."
# 使用 wget 从 Hugging Face URL 下载文件
#wget -O eval/MME/MME_Benchmark_release_version.zip https://alpha.hf-mirror.com/datasets/darkyarding/MME/resolve/main/MME_Benchmark_release_version.zip
#unzip -q eval/MME/MME_Benchmark_release_version.zip -d eval/MME/
#rm eval/MME/MME_Benchmark_release_version.zip
#mv eval/MME/MME_Benchmark_release_version/MME_Benchmark/* eval/MME/MME_Benchmark_release_version/
#rmdir eval/MME/MME_Benchmark_release_version/MME_Benchmark
#wget https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/raw/Evaluation/tools/eval_tool.zip -O eval/MME/eval_tool.zip
#unzip eval/MME/eval_tool.zip -d eval/MME/
#rm eval/MME/eval_tool.zip
echo "==> MME 下载完成。"
echo "============================================="

# MM-Vet 一键安装
#echo "==> 正在下载 MM-Vet 数据集..."
#wget -q --show-progress https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip -P eval/mm-vet/
#unzip -q eval/mm-vet/mm-vet.zip -d eval/mm-vet/
#rm eval/mm-vet/mm-vet.zip
#mv eval/mm-vet/mm-vet/images eval/mm-vet/
echo "==> MM-Vet 下载完成。"
echo "============================================="

# MMMU 需要修改 download_images.py 的下载路径
echo "==> 正在下载 MMMU 数据集 ..."
mkdir -p eval/MMMU
#gdown '1TJszQ23X-7TeMYDA7hVKpoHy9yo-lsc5' -O eval/MMMU/MMMU.zip
#unzip -q eval/MMMU/MMMU.zip -d eval/MMMU/
#mv eval/MMMU/MMMU/* eval/MMMU/
#rmdir eval/MMMU/MMMU/
#rm eval/MMMU/MMMU.zip
echo "==> 正在下载 MMMU 图像..."
echo "------------------------------------------------------------"
echo "重要提示: 根据官方说明, 您可能需要手动编辑"
echo "'eval/MMMU/eval/download_images.py' 文件以设置正确的图像路径。"
echo "请在继续前确认路径设置，或在脚本出错后手动运行此部分。"
echo "------------------------------------------------------------"
# 进入MMMU目录以运行其下载脚本
cd eval/MMMU
mkdir -p all_images
python eval/download_images.py
# 返回基础目录
cd ../..
echo "==> MMMU 下载完成。"
echo "============================================="

echo "所有数据集已成功下载并整理到 '$TARGET_BASE_DIR/eval' 目录中。"