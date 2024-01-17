# 環境構築
```
docker-compose up -d
```
# データの準備
```
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git SGD
python create_original_sgd_data.py 
```
# Trainnig (`finetune.py`)
```
python3.10 finetune.original.py \
--base_model 'baffo32/decapoda-research-llama-7B-hf' \
--data_path './Data/SGD_preprocess/dstc8_all' \
--output_dir './Checkpoint_files/original/SGD_sndc_test_model_server_238' \
--num_epochs=4 \
--cutoff_len=1024 \
--group_by_length \
--lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
--micro_batch_size=16
```
This should take ~30 hours to train on a single Nvidia 3090 GPU. At the end of training, the model's fine-tuned weights will be stored in `$output_dir`.
# Pred
```
python3.10 generate.original.py \
    --load_8bit \
    --base_model 'baffo32/decapoda-research-llama-7B-hf' \
    --lora_weights './Checkpoint_files/original/SGD_test_model2/checkpoint-1250' \
    --testfile_name './Data/SGD_preprocess/dstc8_all' 
```
background
```
nohup python3.10 generate.original.py \
    --load_8bit \
    --base_model 'baffo32/decapoda-research-llama-7B-hf' \
    --lora_weights './Checkpoint_files/original/SGD_test_model2/checkpoint-1250' \
    --testfile_name './Data/SGD_preprocess/dstc8_all' &
```