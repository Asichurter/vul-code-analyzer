from collections import Counter

class DataLabelDistStat:
    def __init__(self, d_list, label_key_extract):
        self.d_list = d_list
        self.label_key_extract = label_key_extract

        self.labels = [label_key_extract(d) for d in d_list]
        self.l_counter = Counter(self.labels)

    def show_dist(self, threshold=1, excluded_keys=['']):
        total = 0
        res = {}
        for k, v in self.l_counter.items():
            if k not in excluded_keys and v >= threshold:
                res[k] = v
                total += v
        print(res)
        print(f'Num of Class: {len(res)}')
        print(f'Class space: {list(res.keys())}')
        print(f'Total instances: {total}')

    def show_dist_from_given_labels(self, labels):
        total = 0
        res = {}
        for k, v in self.l_counter.items():
            if k in labels:
                res[k] = v
                total += v
        print(res)
        print(f'Num of Class: {len(res)}')
        print(f'Total instances: {total}')