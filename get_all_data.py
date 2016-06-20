import csv
import pprint
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
from scipy.signal import argrelextrema


parle_path = 'C:/Users/Goren/Dropbox/Research/CuriosityLab/Research/Parle/'
csv_path = 'AffdexDataFiltered_ConvertFloatToInt_BagsGone/'


def load_basic_data():
    data = {}
    fields = None
    with open(parle_path + 'basic_data.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if not fields:
                fields = row
            else:
                subject = row[0]
                data[subject] = {'valid_fields': set()}
                for k in range(1, len(fields)):
                    if row[k] == 'NaN':
                        data[subject][fields[k]] = None
                    else:
                        data[subject]['valid_fields'].add(fields[k])
                        data[subject][fields[k]] = float(row[k])
    return data, fields


def load_affdex():
    aff_fields = None
    for subject in data.keys():
        day, sub = subject.split('_')
        filename = parle_path + csv_path + 'parle_day' + day + '_subject' + sub + '_filtered.csv'
        # print(filename)
        if os.path.isfile(filename):
            with open(filename , 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                k = 0
                for row in reader:
                    if aff_fields is None and row[0] == 'Smile':
                        aff_fields = ['time']
                        aff_fields.extend(row)
                    if k == 1:
                        data[subject]['affdex'] = []
                    elif k > 1:
                        try:
                            the_data = [float(row[i]) for i in range(0, len(aff_fields)-1)]
                            if sum(abs(np.array(the_data))) > 0:
                                x = [k]
                                x.extend(the_data)
                                data[subject]['affdex'].append(np.array(x))
                            data[subject]['valid_fields'].add('affdex')
                        except:
                            print(subject, 'error in reading csv')
                    k += 1
                data[subject]['affdex'] = np.array(data[subject]['affdex'])
    fields.append('affdex')
    return data, fields, aff_fields


def add_aff_analysis():
    for sub, val in data.items():
        if 'affdex' in val:
            x = val['affdex']
            val['mean_abs_valence'] = np.nanmean(np.abs(x[:, -2]))
            val['mean_engagement'] = np.nanmean(x[:, -1])
            val['std_abs_valence'] = np.nanstd(np.abs(x[:, -2]))
            val['std_valence'] = np.nanstd(x[:, -2])
            val['std_engagement'] = np.nanstd(x[:, -1])

            # find local maxima
            f = argrelextrema(x[:, -1], np.greater)
            val['mean_max_engagement'] = np.nanmean(x[f,-1])

    return data


def get_valid_data(xy_fields):
    x = np.ones([len(data.keys()), 3]) * np.nan
    n = 0
    for sub, val in data.items():
        x[n, 2] = n
        for k in range(0, 2):
            if xy_fields[k] in val:
                x[n, k] = val[xy_fields[k]]
        n += 1

    mask0 = ~np.isnan(x[:,0])
    mask1 = ~np.isnan(x[:,1])
    mask = mask0 & mask1
    x_valid = x[mask, :]
    return x_valid

data, fields = load_basic_data()
data, fields, aff_fields = load_affdex()
data = add_aff_analysis()

xy_fields = [fields[6], 'mean_max_engagement']
x_valid = get_valid_data(xy_fields)
print(x_valid)
plt.plot(x_valid[:, 0], x_valid[:, 1], 'x')
plt.xlabel(xy_fields[0])
plt.ylabel(xy_fields[1])
plt.show()