import os

from utils.file import read_dumped

from allennlp.training.optimizers import Optimizer


base_ver_run_log_cv_path = "/data1/zhijietang/vul_data/run_logs/vul_func_pred/3"
comp_ver_run_log_cv_path = "/data1/zhijietang/vul_data/run_logs/vul_func_pred/4"
cv = 10
metric_key = 'F1-Score'
compare_direction = 1   # 1 or -1

best_delta = float('-inf')
best_cv = None

for i in range(cv):
    base_eval_res_path = os.path.join(base_ver_run_log_cv_path, f'rs_{i}', 'eval_results.json')
    comp_eval_res_path = os.path.join(comp_ver_run_log_cv_path, f'rs_{i}', 'eval_results.json')
    base_result = read_dumped(base_eval_res_path)[-1][metric_key]
    comp_result = read_dumped(comp_eval_res_path)[-1][metric_key]
    metric_delta = (comp_result - base_result) * compare_direction
    print(i, round(metric_delta, 4), round(comp_result, 4), round(base_result, 4))

    if metric_delta > best_delta:
        best_delta = metric_delta
        best_cv = i

print(f'Best CV: #{best_cv}')
print(f'Best {metric_key} delta: {best_delta}')

