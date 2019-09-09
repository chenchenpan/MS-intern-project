DATA_FILE=TenantInfo-and-usage_mature_clip_shuffled.csv
CONFIG_FILE=configure.json
OUTPUT_DIR_ENCODE=/data/home/t-chepan/projects/MS-intern-project/encoded_data_clip_fast/
USE_TEXT_FEATURES=FALSE
OUTPUT_DIR_MODEL=/data/home/t-chepan/projects/MS-intern-project/results_clip_cluster/
TRIAL_NUM=1

# cd ~/projects/MS-intern-project/

# mkdir results_clip_cluster
# mkdir encoded_data_clip_cluster


# python encoding_data.py --data_file $DATA_FILE \
#                         --configure_file $CONFIG_FILE \
#                         --output_dir $OUTPUT_DIR_ENCODE \
#                         --use_text_features $USE_TEXT_FEATURES



python NN-hyp-tuning_2hs_last2ndlayer.py --data_dir $OUTPUT_DIR_ENCODE \
                        --output_dir $OUTPUT_DIR_MODEL \
                        --trial_num $TRIAL_NUM