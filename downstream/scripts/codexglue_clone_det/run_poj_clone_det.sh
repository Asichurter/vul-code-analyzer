# Train
#echo "Train start"
python poj_run.py \
    --output_dir=/data1/zhijietang/vul_data/run_logs/poj_clone_det/4 \
    --model_type=roberta \
    --config_name=/data1/zhijietang/vul_data/transformers_repos/codebert-hybridPDG-mlm-best \
    --model_name_or_path=/data1/zhijietang/vul_data/transformers_repos/codebert-hybridPDG-mlm-best \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=/data1/zhijietang/vul_data/datasets/POJ/clone-det/splits/codexglue_splits/train.jsonl \
    --eval_data_file=/data1/zhijietang/vul_data/datasets/POJ/clone-det/splits/codexglue_splits/valid.jsonl \
    --test_data_file=/data1/zhijietang/vul_data/datasets/POJ/clone-det/splits/codexglue_splits/test.jsonl \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --pooling_func cls \
    --cuda 1 \
    --seed 123456 2>&1 # > /data1/zhijietang/temp/vul_temp/nohup_poj_clone_det/train_ver_2.log &

# Eval
echo "Test start"
python poj_run.py \
    --output_dir=/data1/zhijietang/vul_data/run_logs/poj_clone_det/4 \
    --model_type=roberta \
    --config_name=/data1/zhijietang/vul_data/transformers_repos/codebert-hybridPDG-mlm-best \
    --model_name_or_path=/data1/zhijietang/vul_data/transformers_repos/codebert-hybridPDG-mlm-best \
    --tokenizer_name=roberta-base \
    --do_test \
    --train_data_file=/data1/zhijietang/vul_data/datasets/POJ/clone-det/splits/codexglue_splits/train.jsonl \
    --eval_data_file=/data1/zhijietang/vul_data/datasets/POJ/clone-det/splits/codexglue_splits/valid.jsonl \
    --test_data_file=/data1/zhijietang/vul_data/datasets/POJ/clone-det/splits/codexglue_splits/test.jsonl \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --pooling_func cls \
    --cuda 1 \
    --seed 123456 2>&1 # > /data1/zhijietang/temp/vul_temp/nohup_poj_clone_det/test_ver_1.log &

python poj_evaluator.py -p /data1/zhijietang/vul_data/run_logs/poj_clone_det/4/predictions.jsonl -a /data1/zhijietang/vul_data/datasets/POJ/clone-det/splits/codexglue_splits/test_answers.jsonl