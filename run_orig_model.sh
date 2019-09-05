DATA_FILE=/data/home/t-chepan/projects/MS-intern-project/raw_data/TenantInfo-and-usage_shuffled_inf.csv
CONFIG_FILE=/data/home/t-chepan/projects/MS-intern-project/raw_data/configure.json
ENCODED_DIR=/data/home/t-chepan/projects/MS-intern-project/encoded_data_orig/
USE_TEXT_FEATURES=False
OUTPUT_DIR_MODEL=/data/home/t-chepan/projects/MS-intern-project/results_orig/
TRIAL_NUM=30


# cd ~/project/MS-intern-project
# mkdir encoded_data_orig
# mkdir results_orig

python encoding_data.py --data_file $DATA_FILE \
                        --configure_file $CONFIG_FILE \
                        --output_dir $ENCODED_DIR \
                        --use_text_features $USE_TEXT_FEATURES



python NN-hyp-tuning.py --data_dir $ENCODED_DIR \
                        --output_dir $OUTPUT_DIR_MODEL \
                        --trial_num $TRIAL_NUM