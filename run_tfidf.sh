DATA_FILE=TenantInfo-usage-and-verbatim_mature_clip_shuffled.csv
CONFIG_FILE=configure_wText.json
OUTPUT_DIR_ENCODE=/data/home/t-chepan/projects/MS-intern-project/encoded_data_text/
USE_TEXT_FEATURES=True
OUTPUT_DIR_MODEL=/data/home/t-chepan/projects/MS-intern-project/results_text/
TRIAL_NUM=50

# cd ~/project/MS-intern-project
# rm -rf encoded_data_text
# rm -rf results_text
# mkdir encoded_data_text
# mkdir results_text


# python encoding_data.py --data_file $DATA_FILE \
#                         --configure_file $CONFIG_FILE \
#                         --output_dir $OUTPUT_DIR_ENCODE \
#                         --use_text_features $USE_TEXT_FEATURES



python NN-hyp-tuning.py --data_dir $OUTPUT_DIR_ENCODE \
                        --output_dir $OUTPUT_DIR_MODEL \
                        --trial_num $TRIAL_NUM