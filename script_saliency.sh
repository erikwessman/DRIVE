#!/bin/bash

source activate pyRL

PHASE=$1
GPU_ID=$2
CONFIG=$3

LOG_DIR="./logs/saliency"
if [ ! -d $LOG_DIR ]; then
  mkdir -p -m 777 $LOG_DIR
  echo "mkdir -p -m 777 ${LOG_DIR} done"
fi
rm -rf ${LOG_DIR}/${PHASE}_saliency_*.log

LOG="${LOG_DIR}/${PHASE}_saliency_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

CUDA_VISIBLE_DEVICES=$GPU_IDS

python main_saliency.py \
    --phase $PHASE \
    --gpu_id $GPU_ID \
    --config $CONFIG

echo "Done!"
