from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import os
from pprint import pprint

from utils.file import read_dumped

real_test_file_path = '/data1/zhijietang/vul_data/datasets/reveal/common/random_split/split_2/test.json'
reveal_output_path = '/data1/zhijietang/temp/fse2023_baselines/my_reveal_on_reveal_split2_fixed_res.json'

real_positive, real_negative = 0, 0

real_test_data = read_dumped(real_test_file_path)
for item in real_test_data:
    if item['vul'] == 1:
        real_positive += 1
    else:
        real_negative += 1

def cal_metric_on_results(result_path):
    pred_res = read_dumped(result_path)
    preds, labels = pred_res['preds'], pred_res['labels']

    metrics = {
        'only_model_pred': {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds),
            'recall': recall_score(labels, preds),
            'f1': f1_score(labels, preds),
            'mcc': matthews_corrcoef(labels, preds)
        }
    }

    output_pos = sum(labels)
    output_neg = len(labels) - output_pos
    remain_pos = real_positive - output_pos
    remain_neg = real_negative - output_neg

    preds = preds + [0]*(remain_pos + remain_neg)
    labels = labels + [1]*remain_pos + [0]*remain_neg

    metrics['full'] = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds),
        'mcc': matthews_corrcoef(labels, preds)
    }

    return metrics

def average_metrics_within_folder(metric_folder):
    metrics = []
    for item in os.listdir(metric_folder):
        item_path = os.path.join(metric_folder, item)
        metric = cal_metric_on_results(item_path)
        metrics.append(metric)

    def _avg_metric(_part, _name, _round=2):
        _vals = []
        for _metric in metrics:
            _vals.append(_metric[_part][_name])
        _avg = sum(_vals) / len(_vals) * 100
        if _round is not None:
            return round(_avg, _round)
        else:
            return _avg

    avg_metric = {}
    for on_which_part in ['only_model_pred', 'full']:
        avg_metric[on_which_part] = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'mcc']:
            avg_metric[on_which_part][metric_name] = _avg_metric(on_which_part, metric_name, _round=2)

    print('All metrics: \n')
    pprint(metrics)

    print('\n\n' + '-'*75 + '\n')

    print('Avg metrics: \n')
    pprint(avg_metric)

    # print(avg_metric)

    # print('\n\n' + "-" * 60)
    # print(f'Total items: {len(labels)}\n')
    # print('Overall: ')
    # print(f'accuracy: {accuracy_score(labels, preds)}')
    # print(f'precision: {precision_score(labels, preds)}')
    # print(f'recall: {recall_score(labels, preds)}')
    # print(f'f1: {f1_score(labels, preds)}')
    # print(f'mcc: {matthews_corrcoef(labels, preds)}')

average_metrics_within_folder("/data1/zhijietang/temp/fse2023_baselines/my_reveal_on_reveal_results")