import os
import sys

sys.path.append('/data1/zhijietang/projects/vul-code-analyzer')

from utils.cmd_args import read_aggre_eval_results_args
from utils.file import load_json

def count_mean_metrics(run_log_dir, version, title, cv=5, base_dir='data1'):
    run_log_base_path = f'/{base_dir}/zhijietang/vul_data/run_logs/{run_log_dir}/{version}'
    metric_keys = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score']
    splits = [f'rs_{i}'for i in range(cv)]

    metrics = {k:[] for k in metric_keys}
    for split in splits:
        result_path = os.path.join(run_log_base_path, split, 'eval_results.json')
        # Only load latest eval result item
        try:
            eval_results = load_json(result_path)[-1]
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


if __name__ == '__main__':
    args = read_aggre_eval_results_args()
    count_mean_metrics(args.run_log_dir, args.version, args.title, args.cv)
