#!/bin/bash
PROJ_PATH="/home/path/to/GAGS"
CASE_NAME="garden_scene_name"
DATA_NAME=$(echo $CASE_NAME | cut -d'_' -f1)
if [ $DATA_NAME = "waldo" ]; then
        DATA_NAME="waldo_kitchen"
fi

# setting visualization CAM_ID(invalid in pcd_mode) and PROMPT
CAM_ID="0,1,2"
PROMPT="query1,query2,query3"

python compute_relvancy.py \
        --model_path $PROJ_PATH/output/$CASE_NAME \
        --iteration 30000 \
        --source_path $PROJ_PATH/data/$DATA_NAME \
        --cam_id $CAM_ID \
        --prompt "$PROMPT" \
        --image_mode \
        --video