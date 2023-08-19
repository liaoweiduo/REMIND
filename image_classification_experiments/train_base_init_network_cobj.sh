#!/usr/bin/env bash

PROJ_ROOT=/liaoweiduo/REMIND

export PYTHONPATH=${PROJ_ROOT}
#source activate remind_proj
cd ${PROJ_ROOT}/image_classification_experiments

IMAGENET_DIR=/liaoweiduo/datasets
BASE_MAX_CLASS=10
MODEL=ResNet18ClassifyAfterLayer4_1
LABEL_ORDER_DIR=./imagenet_files/ # location of numpy label files
GPU=0

CUDA_VISIBLE_DEVICES=${GPU} python -u train_base_init_network_from_scratch_cobj.py \
--arch ${MODEL} \
--data ${IMAGENET_DIR} \
--base_max_class ${BASE_MAX_CLASS} \
--labels_dir ${LABEL_ORDER_DIR} \
--ckpt_file ${MODEL}_${BASE_MAX_CLASS}.pth > ../../REMIND-experiments/logs/${MODEL}_${BASE_MAX_CLASS}_from_scratch.log
