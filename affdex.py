import csv
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import math


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
        return idx - 1
    else:
        return idx


parle_path = 'C:/Users/Goren/Dropbox/Research/CuriosityLab/Research/Parle/'
csv_path = 'AffdexDataFiltered_ConvertFloatToInt_BagsGone/'

onlyfiles = [f for f in listdir(parle_path + csv_path) if isfile(join(parle_path + csv_path, f))]
fields = None
data = {}
for f in onlyfiles:
    with open(parle_path + csv_path + f, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        k = 0
        for row in reader:
            if fields is None and row[0] == 'Smile':
                fields = ['time']
                fields.extend(row)
            if k == 1:
                data[f] = {'condition': 0, 'affdex': [], 'length': 0}
            elif k > 1:
                try:
                    the_data = [float(row[i]) for i in range(0, len(fields)-1)]
                    if sum(abs(np.array(the_data))) > 0:
                        x = [k]
                        x.extend(the_data)
                        data[f]['affdex'].append(np.array(x))
                except:
                    print(f, 'error in reading csv')
            k += 1
        data[f]['length'] = len(data[f]['affdex'])
        data[f]['affdex'] = np.array(data[f]['affdex'])

conditions = set()
with open(parle_path + 'Subject Data (Responses).csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        name = 'parle_' + row[1] + '_filtered.csv'
        try:
            data[name]['condition'] = row[2]
            conditions.add(row[2])
        except:
            pass

print(fields)
print('number of subjects: ', len(data.keys()))
print('conditions: ', conditions)
print(data)
# analysis
analysis = {}

# percentiles
perc = np.array([0, 0.25, 0.50, 0.75, 1.00]) * 100
# perc = np.array([0, 0.50, 1.00]) * 100
# perc = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00]) * 100
# perc = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.00]) * 100
for c in conditions:
    analysis[c] = {'percetile averages': {'raw': [], 'stats': []}}

for k, v in data.items():
    c = v['condition']
    x = v['affdex']

    percentiles = np.round(np.percentile(x[:, 0], perc))
    y = np.zeros([percentiles.shape[0] - 1, len(fields)])
    for f in range(0, len(fields)):
        for p in range(1, percentiles.shape[0]):
            t0 = find_nearest(x[:, 0], percentiles[p-1])
            t1 = find_nearest(x[:, 0], percentiles[p])
            # y[p-1, f] = np.mean(x[t0:t1, f])
            x_zero = x[t0:t1, f]
            x_zero[x_zero == 0] = np.nan
            y[p-1, f] = np.nanmean(x_zero)
    analysis[c]['percetile averages']['raw'].append(y)

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
ax = axs
counts = {}
for c, v in analysis.items():
    x = np.array(v['percetile averages']['raw'])[:, :, 1:]
    np.savetxt(c + '_data.csv', x[:,:,-1], delimiter=',')
    if 'Tablet' not in c:
        print(c, x.shape)
        diff = (x[:,-1,-1] - x[:,0,-1])
        mean_diff = np.average(diff, axis=0)
        std_diff = np.std(diff, axis=0)
        print(mean_diff)
        print(std_diff / np.sqrt(x.shape[0]))

        mean_eng = np.average(x[:,:,-1], axis=0)
        std_eng = np.std(x[:,:,-1], axis=0)
        sem_eng = std_eng / np.sqrt(x.shape[0])
        ax.errorbar(perc[1:],mean_eng , sem_eng , fmt='-o', label=c)

        if 'Curious' in c:
            stat_x = diff
        if 'Noncurious' in c:
            stat_y = diff


ax.set_xlim([0, 125])
ax.set_ylim([0, 50])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
# plt.show()
print(counts)
# export = np.zeros()
# for c, v in analysis.items():
#     x = np.array(v['percetile averages']['raw'])[:, :, 1:]
#     if 'Tablet' not in c:
