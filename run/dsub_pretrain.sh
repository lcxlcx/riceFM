#!/bin/bash
#DSUB -n scgpt_pretrain_rice
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp
#DSUB -R "cpu=54;gpu=4;mem=50000"
#DSUB -oo ./logs/scgpt_rice_pretrain.out
#DSUB -eo ./logs/scgpt_rice_pretrain.err

## Set scripts
RANK_SCRIPT="pretrain.sh"

###Set Start Path
JOB_PATH="/home/share/huadjyin/home/s_qiuping1/workspace/plant/scGPT/run/"

## Set NNODES
NNODES=1

## Create nodefile

JOB_ID=${BATCH_JOB_ID}
NODEFILE=/home/share/huadjyin/home/s_qiuping1/workspace/plant/scGPT/run/logs/${JOB_ID}.nodefile

touch $NODEFILE

cd ${JOB_PATH};/usr/bin/bash ${RANK_SCRIPT} ${NNODES} ${NODEFILE}