import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_base_transformation(s):
    aux = s

    l = aux.find(' - ')
    aux = aux[:l]
    
    while True:
        l = aux.find('(')
        r = aux.find(')')
        
        if l == -1: break
    
        aux = aux[:l] + aux[r+1:]
        
    return aux


def clean_ts(v):
    v = v[:v.find(' - ')] if v.find(' - ') != -1 else v
    return v.replace(' watermarking', '').replace('message error for ','').replace('success rate for ' ,'')


def get_rank(x, y):
    v = []

    for i in range(len(x)):
        v.append((chr(ord('a') + int(x[i])) if x[i] < 3 else ' ') + (chr(ord('a') + int(y[i])) if y[i] < 3 else ' '))
    
    return v


data_file = sorted([f for f in os.listdir() if f[-4:] == '.pkl'], reverse=True)[0]

with open(data_file,'rb') as f:
    data = pickle.load(f)
    
coins = list(data.keys())
methods = list(data[coins[0]].keys())
transformations = list(data[coins[0]][methods[0]]['validation'].keys())

grouped_transformations = {}
for transformation in transformations:
    aux = get_base_transformation(transformation)

    if aux in grouped_transformations:
        grouped_transformations[aux] += 1
    else:
        grouped_transformations[aux] = 1
    
kind_transformation_sets = {
    'geometric': transformations[1:37] + transformations[61:65],
    'photometric': transformations[37:61] + transformations[65:],
    'all': transformations[1:],
    }

kind_transformations = {}
for k in kind_transformation_sets.keys():
    kind_transformations[k] = {}
    aux = kind_transformation_sets[k]
    for kk in aux:
        kind_transformations[k][kk] = 1
    
table = {}
for method in methods:
    row = {}

    row['psnr'] = 0
    row['ssim'] = 0

    row['failures'] = 0

    row['error'] = {}
    row['pass'] = {}

    for transformation in transformations:
        row['error'][transformation] = 0
        row['pass'][transformation] = 0

    for transformation in grouped_transformations.keys():
        row['error'][transformation] = 0
        row['pass'][transformation] = 0

    for transformation in kind_transformation_sets.keys():
        row['error'][transformation] = 0
        row['pass'][transformation] = 0

    for coin in coins:
        if not (method in data[coin]):
            row['failures'] += 1            
            continue

        if 'PSNR' in data[coin][method]:     
            row['psnr'] += data[coin][method]['PSNR']

        if 'SSIM' in data[coin][method]:       
            row['ssim'] += data[coin][method]['SSIM']
            
        for transformation in transformations:
            what = data[coin][method]['validation'][transformation]['correct bits']
            row['error'][transformation] += what
            row['pass'][transformation] += (what == 1)
            
            group_transformation = get_base_transformation(transformation)
            row['error'][group_transformation] += what / grouped_transformations[group_transformation]
            row['pass'][group_transformation] += (what == 1) / grouped_transformations[group_transformation]

            for k in kind_transformation_sets.keys():
                if transformation in kind_transformations[k]:
                    row['error'][k] += what / len(kind_transformations[k])
                    row['pass'][k] += (what == 1) / len(kind_transformations[k])
                                                
    for el in ['psnr', 'ssim', 'failures']:
        row[el] /= len(coins)

    for el in ['error', 'pass']:
        for transformation in transformations:
            row[el][transformation] /= len(coins)

        for transformation in grouped_transformations.keys():
            row[el][transformation] /= len(coins)

        for k in kind_transformation_sets.keys():
            row[el][k] /= len(coins)

    table[method] = row
    
r = len(methods)

###

c1 = ['psnr', 'ssim', 'failures']
m1 = np.zeros((r, len(c1)))

for i in range(r):
    for j in range(len(c1)):
        m1[i, j] = table[methods[i]][c1[j]]

v1 = {'method': [m.replace(' watermarking+',' + ') for m in methods]}
for i, v in enumerate(['psnr', 'ssim']): v1[v] = m1[:, i]
pd.DataFrame(v1).to_csv('visual quality.csv', sep=';', float_format=lambda s: "{: 6.2f}".format(s*100), index=False)       

###
    
c2 = transformations + list(grouped_transformations.keys()) + list(kind_transformation_sets.keys())
m2 = np.zeros((r, len(c2)))

for i in range(r):
    for j in range(len(c2)):
        m2[i, j] = table[methods[i]]['error'][c2[j]]

vv2 = [v[:v.find(' - ')] if v.find(' - ') != -1 else v for v in c2]
vv2 = [v.replace(' watermarking', '') for v in vv2]
v2 = {'method': [m.replace(' watermarking+',' + ') for m in methods]}
for i, v in enumerate(c2): v2[vv2[i]] = m2[:, i]
pd.DataFrame(v2).to_csv('message error.csv', sep=';', float_format=lambda s: "{: 6.2f}".format(s*100), index=False)       

###

c3 = transformations + list(grouped_transformations.keys()) + list(kind_transformation_sets.keys())
m3 = np.zeros((r, len(c3)))

for i in range(r):
    for j in range(len(c3)):
        m3[i, j] = table[methods[i]]['pass'][c3[j]]

vv3 = [v[:v.find(' - ')] if v.find(' - ') != -1 else v for v in c3]
vv3 = [v.replace(' watermarking', '') for v in vv3]
v3 = {'method': [m.replace(' watermarking+',' + ') for m in methods]}
for i, v in enumerate(c3): v3[vv3[i]] = m3[:, i]
pd.DataFrame(v3).to_csv('success rate.csv', sep=';', float_format=lambda s: "{: 6.2f}".format(s*100), index=False)       

###

r_labels = methods
c_labels = c1 + ['message error for ' + v for v in c2] + ['success rate for ' + v for v in c3]
m = np.concatenate((m1, m2, m3), axis=1)

m_idx = np.zeros(m.shape)
for i in range(m.shape[1]):
    _, _, aux= np.unique(-m[:, i], return_index=True, return_inverse=True)
    m_idx[:, i] = aux


single_method = np.array([True if v.find('+') == -1 else False for v in methods])
m_idx_ = np.zeros(m.shape)

for i in range(m.shape[1]):
    _, _, aux= np.unique(-m[single_method, i], return_index=True, return_inverse=True)
    m_idx_[single_method, i] = aux

    _, _, aux= np.unique(-m[~single_method, i], return_index=True, return_inverse=True)
    m_idx_[~single_method, i] = aux

###

c_labels = c_labels[:2] + c_labels[105:125] + c_labels[227:] 
m = np.concatenate((m[:, :2], m[:, 105:125], m[:, 227:]), axis=1) 
m_idx = np.concatenate((m_idx[:, :2], m_idx[:, 105:125], m_idx[:, 227:]), axis=1) 
m_idx_ = np.concatenate((m_idx_[:, :2], m_idx_[:, 105:125], m_idx_[:, 227:]), axis=1) 

row_to_retain = np.any((m_idx[:, [19, 20 , 21, 39, 40, 41]] < 3) | (m_idx_[:, [19, 20 , 21, 39, 40, 41]] < 3), axis=1)
row_to_retain[:8] = True

r_labels = [r_labels[i] for i, v in enumerate(row_to_retain) if v]
m = m[row_to_retain]
m_idx = m_idx[row_to_retain]
m_idx_ = m_idx_[row_to_retain]

###

v1 = {'method': [m.replace(' watermarking+',' + ') for m in r_labels]}
for i in range(2): v1[c_labels[i]] = m[:, i]
pd.DataFrame(v1).to_csv('visual quality (reduced table).csv', sep=';', float_format=lambda s: "{: 6.2f}".format(s*100), index=False)       

v2 = {'method': [m.replace(' watermarking+',' + ') for m in r_labels]}
for i in range(2, 22): v2[clean_ts(c_labels[i])] = m[:, i]
for i in range(19, 22): v2[clean_ts(c_labels[i]) + ' (rank)' ] = get_rank(m_idx[:, i],m_idx_[:, i])
pd.DataFrame(v2).iloc[:, list(range(0, 18)) + [18, 21, 19, 22, 20, 23]].to_csv('message error (reduced table).csv', sep=';', float_format=lambda s: "{: 6.2f}".format(s*100), index=False)       

v3 = {'method': [m.replace(' watermarking+',' + ') for m in r_labels]}
for i in range(22, 42): v3[clean_ts(c_labels[i])] = m[:, i]
for i in range(39, 42): v3[clean_ts(c_labels[i]) + ' (rank)' ] = get_rank(m_idx[:, i],m_idx_[:, i])
pd.DataFrame(v3).iloc[:, list(range(0, 18)) + [18, 21, 19, 22, 20, 23]].to_csv('success rate (reduced table).csv', sep=';', float_format=lambda s: "{: 6.2f}".format(s*100), index=False)       

###