#!/usr/bin/env bash

PROJ_ROOT=/liaoweiduo/REMIND

export PYTHONPATH=${PROJ_ROOT}
#source activate remind_proj
cd ${PROJ_ROOT}/image_classification_experiments

#IMAGE_DIR=/media/tyler/nvme_drive/data/ImageNet2012
IMAGE_DIR=/liaoweiduo/datasets
EXPT_NAME=remind_cgqa_lr0_1
RESUME_FULL_PATH=/liaoweiduo/REMIND-experiments/${EXPT_NAME}/remind_model
GPU=7

REPLAY_SAMPLES=50
MAX_BUFFER_SIZE=959665
CODEBOOK_SIZE=256
NUM_CODEBOOKS=32
BASE_INIT_CLASSES=10
CLASS_INCREMENT=10
NUM_CLASSES=110
STREAMING_MIN_CLASS=10
STREAMING_MAX_CLASS=100
#BASE_INIT_CKPT=./imagenet_files/best_ResNet18ClassifyAfterLayer4_1_100.pth # base init ckpt file
LABEL_ORDER_DIR=./imagenet_files/ # location of numpy label files

CUDA_VISIBLE_DEVICES=${GPU} python -u CFST_experiment.py \
--images_dir ${IMAGE_DIR} \
--datasets CGQA \
--max_buffer_size ${MAX_BUFFER_SIZE} \
--num_classes ${NUM_CLASSES} \
--streaming_min_class ${STREAMING_MIN_CLASS} \
--streaming_max_class ${STREAMING_MAX_CLASS} \
--base_init_classes ${BASE_INIT_CLASSES} \
--class_increment ${CLASS_INCREMENT} \
--classifier ResNet18_StartAt_Layer4_1 \
--rehearsal_samples ${REPLAY_SAMPLES} \
--start_lr 0.5 \
--end_lr 0.001 \
--lr_step_size 100 \
--lr_mode cos \
--weight_decay 1e-5 \
--use_random_resized_crops \
--use_mixup \
--mixup_alpha .1 \
--label_dir ${LABEL_ORDER_DIR} \
--num_codebooks ${NUM_CODEBOOKS} \
--codebook_size ${CODEBOOK_SIZE} \
--expt_name ${EXPT_NAME} > ../../REMIND-experiments/logs/${EXPT_NAME}.log

#--classifier_ckpt ${BASE_INIT_CKPT} \
# --resume_full_path ${RESUME_FULL_PATH} \
