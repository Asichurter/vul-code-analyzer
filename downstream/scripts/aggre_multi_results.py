import os
import sys

sys.path.append('/data1/zhijietang/projects/vul-code-analyzer')

from utils.cmd_args import read_aggre_eval_results_args
from utils.file import load_json

args = read_aggre_eval_results_args()

run_log_base_path = f'/data1/zhijietang/vul_data/run_logs/{args.run_log_dir}/{args.version}'
metric_keys = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score']
title = args.title
splits = ['rs_0', 'rs_1', 'rs_2', 'rs_3', 'rs_4']

metrics = {k:[] for k in metric_keys}
for split in splits:
    result_path = os.path.join(run_log_base_path, split, 'eval_results.json')
    # Only load first eval result item
    try:
        eval_results = load_json(result_path)[0]
    except FileNotFoundError:
        print(f'# {split} not found, skip')
        continue

    for key in metric_keys:
        try:
            metrics[key].append(eval_results[key])
        except KeyError as e:
            continue

print(f'Results for {title}')
print('*'*60)
for key in metrics:
    if len(metrics[key]) == 0:
        continue
    print(f'Avg {key}: {sum(metrics[key]) / len(metrics[key])}')
print('*'*60)
