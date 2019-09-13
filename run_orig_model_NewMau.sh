DATA_FILE=TenantInfo-and-usage_wNewMAU_mature_clip_shuffled.csv
CONFIG_FILE=configure_wNewMAU.json
OUTPUT_DIR_ENCODE=/data/home/t-chepan/projects/MS-intern-project/encoded_data_NewMAU/
USE_TEXT_FEATURES=False
OUTPUT_DIR_MODEL=/data/home/t-chepan/projects/MS-intern-project/results_NewMAU/
TRIAL_NUM=50

cd ~/project/MS-intern-project
mkdir encoded_data_NewMAU
mkdir results_NewMAU


python encoding_data.py --data_file $DATA_FILE \
                        --configure_file $CONFIG_FILE \
                        --output_dir $OUTPUT_DIR_ENCODE \
                        --use_text_features $USE_TEXT_FEATURES



python NN-hyp-tuning_faster.py --data_dir $OUTPUT_DIR_ENCODE \
                        --output_dir $OUTPUT_DIR_MODEL \
                        --trial_num $TRIAL_NUM