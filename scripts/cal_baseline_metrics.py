import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# result_paths = ['/data1/zhijietang/temp/vul_temp/vuldeepecker_reveal_results.txt',
#                 '/data1/zhijietang/temp/vul_temp/vuldeepecker_devign_results.txt',
#                 '/data1/zhijietang/temp/vul_temp/vuldeepecker_fan_results.txt']

base_dir_path = '/data1/zhijietang/temp/fse2023_baselines/exp2/Reveal'
result_paths = [os.path.join(base_dir_path, item) for item in os.listdir(base_dir_path)]
print(f'Model: {base_dir_path.split("/")[-1]}')

for result_path in result_paths:
    preds, labels = [], []
    with open(result_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        label, pred = line.split()
        labels.append(int(float(label)))
        preds.append(int(float(pred)))

    title = result_path.split('/')[-1]
    print('\n\n' + "-"*60)
    print(f'Total items: {len(labels)}\n')
    print(f'{title}:')
    print(f'accuracy: {accuracy_score(labels, preds)}')
    print(f'precision: {precision_score(labels, preds)}')
    print(f'recall: {recall_score(labels, preds)}')
    print(f'f1: {f1_score(labels, preds)}')
    print(f'mcc: {matthews_corrcoef(labels, preds)}')