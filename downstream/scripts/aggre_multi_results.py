import os

from utils.cmd_args import read_aggre_eval_results_args
from utils.file import load_json

args = read_aggre_eval_results_args()

run_log_base_path = f'/data1/zhijietang/vul_data/run_logs/{args.run_log_dir}/'
versions = args.versions.split(',')
metric_keys = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score']
title = args.title

metrics = {k:[] for k in metric_keys}
for ver in versions:
    result_path = os.path.join(run_log_base_path, str(ver), 'eval_results.json')
    # Only load first eval result item
    eval_results = load_json(result_path)[0]

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
