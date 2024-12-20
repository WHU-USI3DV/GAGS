
PROJ_PATH="/home/path/to/GAGS"
DATA_NAME="garden"
CASE_NAME="garden_scene_name"
Iteration="30000" # iteration of the pretrained RGB 3DGS scene

echo "Running Granularity-Aware Segmentation & Feature Extraction on $CASE_NAME"

if [ -d "$PROJ_PATH/output/$CASE_NAME" ] && [ "$(ls -A $PROJ_PATH/output/$CASE_NAME)" ]; then
    echo "Using pretrained 3DGS scene in $PROJ_PATH/output/$CASE_NAME"
else 
    echo "No pretrained 3DGS scene founded in $PROJ_PATH/output/$CASE_NAME. Please checking again."
    exit 1
fi

if [ -d "$PROJ_PATH/output/$CASE_NAME/train/ours_$Iteration/depths" ] && 
[ "$(ls -A $PROJ_PATH/output/$CASE_NAME/train/ours_$Iteration/depths)" ]; then
    echo "Find rendering depth in $CASE_NAME/train/ours_$Iteration/depths. Skip."
else
    echo "No rendering depth founded in $CASE_NAME/train/ours_$Iteration/depths. Start rendering depth."
    python render.py \
        --source_path $PROJ_PATH/data/$DATA_NAME \
        --model_path $PROJ_PATH/output/$CASE_NAME \
        --iteration $((Iteration)) \
        --render_mode "RGB+ED" \
        --foundation_model "none"
fi

if [ -d "$PROJ_PATH/data/$DATA_NAME/depths_sample" ] &&
[ "$(ls -A $PROJ_PATH/data/$DATA_NAME/depths_sample)" ]; then
    echo "Find min-depth mapping in $PROJ_PATH/data/$DATA_NAME/depths_sample. Skip."
else
    echo "No min-depth mapping founded in $PROJ_PATH/data/$DATA_NAME/depths_sample. Start calculating."
    python depth_SAM.py \
        --source_path $PROJ_PATH/data/$DATA_NAME \
        --model_path $PROJ_PATH/output/$CASE_NAME \
        --foundation_model "none"
fi

if [ -d "$PROJ_PATH/data/$DATA_NAME/language_features" ] &&
[ "$(ls -A $PROJ_PATH/data/$DATA_NAME/language_features)" ]; then
    echo "Find language features in $PROJ_PATH/data/$DATA_NAME/language_features. Skip."
else
    echo "No language features founded in $PROJ_PATH/data/$DATA_NAME/language_features. Start extracting language features."
    python preprocess.py \
        --dataset_path $PROJ_PATH/data/$DATA_NAME \
        --model_path $PROJ_PATH/output/$CASE_NAME \
        --iteration $((Iteration)) \
        --mindepth_mode
    echo "Extracting language features done."
fi