
PROJ_PATH="/home/path/to/GAGS"
DATA_NAME="garden"
CASE_NAME="garden_case_name"
start_checkpoint="chkpnt30000.pth" # pretrained RGB 3DGS scene
Iteration="30000" # iteration of the feature distillation


python train.py \
    -s $PROJ_PATH/data/$DATA_NAME \
    -m $PROJ_PATH/output/$CASE_NAME \
    --start_checkpoint $PROJ_PATH/output/$CASE_NAME/$start_checkpoint \
    -r 2 \
    --iterations $((Iteration)) \
    --feature_mode 
