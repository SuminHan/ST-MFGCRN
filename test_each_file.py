import numpy as np
import os
import pickle
with open('process-qt-CPT-190301-190501-190601-190701.pkl', 'rb') as f:
    mdata = pickle.load(f)

def metric(pred, label):
    mae = np.abs(np.subtract(pred, label)).astype(np.float32)
    rmse = np.square(mae)
    mae = np.mean(mae)
    rmse = np.sqrt(np.mean(rmse))
    nmae = np.sum(np.abs(np.subtract(pred, label))) / np.sum(np.abs(label))
    return mae, rmse, nmae

def metric_print(pred, label):
    mae, rmse, nmae = metric(pred, label)
    return print(f"{mae:.4f}\t{rmse:.4f}\t{nmae:.4f}")

def metric_str(pred, label):
    mae, rmse, nmae = metric(pred, label)
    return f"{mae:.4f}\t{rmse:.4f}\t{nmae:.4f}"

pred_trend = pred = mdata['test']['XQ'].mean(axis=1)
label = mdata['test']['Y']
print('Trend Mean', metric_str(pred, label), sep='\t')

pred =  mdata['test']['XP'].mean(axis=1)
label = mdata['test']['Y']
print('Period Mean', metric_str(pred, label), sep='\t')

pred = mdata['test']['XC'].mean(axis=1)
label = mdata['test']['Y']
print('Closeness Mean', metric_str(pred, label), sep='\t')

pred = mdata['test']['XC'][:, -1, :, :]
label = mdata['test']['Y']
print('Last Repeat', metric_str(pred, label), sep='\t')


pdir = 'test/process-qt-CPT-190301-190501-190601-190701.pkl'

label = np.load(pdir + '/label.npy')

for fname in sorted(os.listdir(pdir)):
    if 'pred' in fname:
        pred_tmp = np.load(pdir + '/' + fname)
        print(fname[5:-4], metric_str(pred_tmp, label), sep='\t')