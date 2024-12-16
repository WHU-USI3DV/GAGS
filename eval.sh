#!/bin/bash
PROJ_PATH="/home/path/to/GAGS"
CASE_NAME="garden_scene_name"
DATA_NAME=$(echo $CASE_NAME | cut -d'_' -f1)
if [ $DATA_NAME = "waldo" ]; then
        DATA_NAME="waldo_kitchen"
fi
GT_FOLDER="data/label" # path to json GT label file

echo "Running Python script with case: $CASE_NAME"
python evaluate_iou_loc.py \
        --dataset_name $DATA_NAME \
        --model_path $PROJ_PATH/output/$CASE_NAME \
        --mask_thresh 0.4 \
        --json_folder $GT_FOLDER \
        --iteration 30000 \
        --source_path $PROJ_PATH/data/$DATA_NAME
        


