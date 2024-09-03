python ../src/evaluation/evaluate.py \
--gpt_type gpt-35-turbo-16k \
--model_name gpt-35-turbo-16k \
--eval_method multichoice \
--input_path ../output \
--file_name ours_multichoice_gpt-35-turbo-16k_EHRNoteQA_processed.csv  \
--save_path ../data
