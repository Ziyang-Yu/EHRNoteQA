python ../src/evaluation/evaluate.py \
--gpt_type gpt-35-turbo \
--model_name Llama-2-7b-chat-hf \
--eval_method multichoice \
--input_path ../output \
--file_name ../output/ours_multichoice_Llama-2-7b-chat-hf_.csv \
--save_path ../output 
