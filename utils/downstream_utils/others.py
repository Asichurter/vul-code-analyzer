

def print_cvss_metrics(metrics):
    new_metrics = {}
    for k,v in metrics.items():
        k = k.replace('_score', '')
        new_metrics[k] = round(v, 4)
    print(new_metrics)