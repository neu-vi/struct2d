#!/bin/bash
#SBATCH --job-name=scannet_all_wkf
#SBATCH --output=/scratch/zhu.fang/jobs/scannet_all_wkf.out
#SBATCH --time=24:00:00
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48    # You request 10 cores here
#SBATCH --mem=196G

conda activate struct2d

export HF_HOME=/scratch/zhu.fang/cache
export TRITON_CACHE_dir=/scratch/zhu.fang/triton_cache
CUDA_VISIBLE_DEVICES=0,1,2,3
llamafactory-cli train configs/train_full/qwen2_5vl_3B_full_sft_tune_all_scannet_all_wkf.yaml