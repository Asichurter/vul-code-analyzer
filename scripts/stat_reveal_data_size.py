import numpy

from utils.file import load_json

vul_path = '/data1/zhijietang/vul_data/datasets/reveal/vulnerables.json'
non_vul_path = '/data1/zhijietang/vul_data/datasets/reveal/non-vulnerables.json'
vul_lens = []
non_vul_lens = []

# def make_hist_probs(hist):
#

vul_data = load_json(vul_path)
non_vul_data = load_json(non_vul_path)

for data in vul_data:
    vul_lens.append(data['size'])

for data in non_vul_data:
    non_vul_lens.append(data['size'])

vul_hist = numpy.histogram(vul_lens, range=(0,500), bins=10)
non_vul_hist = numpy.histogram(non_vul_lens, range=(0,500), bins=10)

lens = vul_lens + non_vul_lens
hist = numpy.histogram(lens, range=(0,500), bins=10)
probs = []

for i in range(len(hist[0])):
    probs.append(hist[0][i] * 1.0 / len(lens))

# for i in range(len(vul_hist[0])):
#     vul_hist[0][i] = vul_hist[0][i] * 1.0 / len(vul_data)

