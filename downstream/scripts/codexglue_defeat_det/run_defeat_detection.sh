# Train
echo "Train start"
python devign_run.py \
    --output_dir=/data1/zhijietang/vul_data/run_logs/devign/10 \
    --model_type=roberta \
    --tokenizer_name=/data1/zhijietang/vul_data/transformers_repos/codebert-hybridPDG-mlm \
    --model_name_or_path=/data1/zhijietang/vul_data/transformers_repos/codebert-hybridPDG-mlm \
    --do_train \
    --train_data_file=/data1/zhijietang/vul_data/datasets/devign/codex_glue_splits/github_split/train.jsonl \
    --eval_data_file=/data1/zhijietang/vul_data/datasets/devign/codex_glue_splits/github_split/valid.jsonl \
    --test_data_file=/data1/zhijietang/vul_data/datasets/devign/codex_glue_splits/github_split/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --cuda 1 \
    --seed 123456  2>&1 # > /data1/zhijietang/temp/vul_temp/nohup_devign/devign_ver_10_train.log &

# Eval
echo "Test start"
python devign_run.py \
    --output_dir=/data1/zhijietang/vul_data/run_logs/devign/10 \
    --model_type=roberta \
    --tokenizer_name=/data1/zhijietang/vul_data/transformers_repos/codebert-hybridPDG-mlm \
    --model_name_or_path=/data1/zhijietang/vul_data/transformers_repos/codebert-hybridPDG-mlm \
    --do_eval \
    --do_test \
    --train_data_file=/data1/zhijietang/vul_data/datasets/devign/codex_glue_splits/github_split/train.jsonl \
    --eval_data_file=/data1/zhijietang/vul_data/datasets/devign/codex_glue_splits/github_split/valid.jsonl \
    --test_data_file=/data1/zhijietang/vul_data/datasets/devign/codex_glue_splits/github_split/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --cuda 1 \
    --seed 123456 2>&1 # > /data1/zhijietang/temp/vul_temp/nohup_devign/devign_ver_10_test.log &

# Result
python devign_evaluator.py -p /data1/zhijietang/vul_data/run_logs/devign/10/predictions.txt -a /data1/zhijietang/vul_data/datasets/devign/codex_glue_splits/github_split/test.jsonl