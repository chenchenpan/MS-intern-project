DATA_FILE=TenantInfo-and-usage_shuffled_inf_clip_mature.csv
CONFIG_FILE=configure.json
OUTPUT_DIR_ENCODE=/data/home/t-chepan/projects/MS-intern-project/encoded_data_clip_1hot/
USE_TEXT_FEATURES=FALSE
OUTPUT_DIR_MODEL=/data/home/t-chepan/projects/MS-intern-project/results_clip_1hot/
TRIAL_NUM=50

# cd ~/projects/MS-intern-project/

# mkdir results_clip_1hot
# mkdir encoded_data_clip_1hot


# python encoding_data_woNormalize1HotFeatures.py --data_file $DATA_FILE \
#                         --configure_file $CONFIG_FILE \
#                         --output_dir $OUTPUT_DIR_ENCODE \
#                         --use_text_features $USE_TEXT_FEATURES



python NN-hyp-tuning_faster.py --data_dir $OUTPUT_DIR_ENCODE \
                        --output_dir $OUTPUT_DIR_MODEL \
                        --trial_num $TRIAL_NUM