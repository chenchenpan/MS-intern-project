DATA_DIR=/data/home/t-chepan/projects/MS-intern-project/encoded_data_clip_fast/
EXPERI_DIR=/data/home/t-chepan/projects/MS-intern-project/results_clip_fast/
# OUTPUT_DIR=/data/home/t-chepan/projects/MS-intern-project/results_clip_fast/
TRIAL_NUM=10



python Permutation-Feature-Importance-Analysis.py --data_dir $DATA_DIR \
                                                  --experiment_dir $EXPERI_DIR \
                                                  --output_dir $EXPERI_DIR \
                                                  --trial_num $TRIAL_NUM