import os
import h5py
import logging
import mxnet as mx
import numpy as np
import pandas as pd
import math
from os.path import join as pjoin

from data import utils
# from config import DATA_PATH, NUM_NODES

# def get_geo_feature_file(dataset):
# 	geo = utils.load_h5(os.path.join(DATA_PATH, 'GN_FEATURE_POI316.h5'), ['embeddings'])
# 	#row, col, _ = geo.shape
# 	#geo = np.reshape(geo, (row * col, -1))

# 	geo = (geo - np.mean(geo, axis=0)) / (np.std(geo, axis=0) + 1e-8)	
# 	return geo

def distant_matrix(n_neighbors, extdata):
	dist = extdata['dist_arr']
	n = dist.shape[0]

	e_in, e_out = [], []
	for i in range(n):
		e_in.append(np.argsort(dist[:, i])[:n_neighbors + 1])
		e_out.append(np.argsort(dist[i, :])[:n_neighbors + 1])
	e_in = np.array(e_in, dtype=np.int32)
	e_out = np.array(e_out, dtype=np.int32)

	adj_mat = np.array(dist)
	e_in = np.array(e_in)
	e_out = np.array(e_out)
	return adj_mat, e_in, e_out

def get_geo_feature(dataset, extdata):
    # get locations
    n_neighbors = dataset['n_neighbors']
    loc = extdata['LOC']
    loc = (loc - np.mean(loc, axis=0)) / np.std(loc, axis=0)

    # get distance matrix
    dist, e_in, e_out = distant_matrix(n_neighbors, extdata)

    # normalize distance matrix
    n = loc.shape[0]
    edge = np.zeros((n, n))
    for i in range(n):
        for j in range(n_neighbors):
            edge[e_in[i][j], i] = edge[i, e_out[i][j]] = 1
    dist[edge == 0] = np.inf

    values = dist.flatten()
    values = values[values != np.inf]
    dist_mean = np.mean(values)
    dist_std = np.std(values)
    dist = np.exp(-(dist - dist_mean) / dist_std)

    # merge features
    features = []
    if extdata['feat'] != 'None':
        extfeat = extdata[extdata['feat']]
        extfeat = extfeat / (extfeat.max(0) + 1e-1)

    for i in range(n): # f = np.concatenate([loc[i], dist[e_in[i], i], dist[i, e_out[i]]])
        if extdata['feat'] == 'None':
            f = np.concatenate([loc[i], dist[e_in[i], i], dist[i, e_out[i]]])
        else:
            f = np.concatenate([loc[i],  extfeat[i] ])
        features.append(f)
    features = np.stack(features)
    return features, (dist, e_in, e_out)

# def dataloader(dataset, args):
# 	data = pd.read_hdf(pjoin(args.data_dir, f'{args.region}/traffic.h5'))
	
# 	n_timestamp = data.shape[0]

# 	num_train = round(n_timestamp * dataset['train_prop'])
# 	num_eval = round(n_timestamp * dataset['eval_prop'])
# 	num_test = n_timestamp - num_train - num_eval

# 	train = data.iloc[:num_train].copy()
# 	eval = data.iloc[num_train: num_train + num_eval].copy()
# 	test = data.iloc[-num_test:].copy()

# 	return train, eval, test


def dataiter_all_sensors_seq2seq(df, scaler, setting, args, shuffle=True):
	dataset = setting['dataset']
	training = setting['training']

	# df_fill = utils.fill_missing(df)
	df_fill = scaler.transform(df)
	NUM_NODES = df.shape[1]
	print('NUM_NODES:', NUM_NODES)

	n_timestamp = df_fill.shape[0]
	data_list = [np.expand_dims(df_fill.values, axis=-1)]

	# time in day
	time_idx = (df_fill.index.values - df_fill.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
	time_in_day = np.tile(time_idx, [1, NUM_NODES, 1]).transpose((2, 1, 0))
	data_list.append(time_in_day)

	# day in week
	time_ind = df.index.weekday
	dayofweek = np.tile(time_ind, [1, NUM_NODES, 1]).transpose((2, 1, 0))
	data_list.append(dayofweek)
	
	# temporal feature
	temporal_feature = np.concatenate(data_list, axis=-1)

	print('temporal_feature: ', temporal_feature.shape)

	geo_feature, _ = get_geo_feature(dataset, args)

	input_len = dataset['input_len']
	output_len = dataset['output_len']
	feature, data, mask, label  = [], [], [], []
	for i in range(n_timestamp - input_len - output_len + 1):
		data.append(temporal_feature[i: i + input_len])

		_mask = np.array(df.iloc[i + input_len: i + input_len + output_len] > 1e-5, dtype=np.float32)
		mask.append(_mask)

		label.append(temporal_feature[i + input_len: i + input_len + output_len])
		
		feature.append(geo_feature)

		if i % 1000 == 0:
			logging.info('Processing %d timestamps', i)

	data = mx.nd.array(np.stack(data))
	label = mx.nd.array(np.stack(label))
	mask = mx.nd.array(np.expand_dims(np.stack(mask), axis=3))
	feature = mx.nd.array(np.stack(feature))

	logging.info('shape of feature: %s', feature.shape)
	logging.info('shape of data: %s', data.shape)
	logging.info('shape of mask: %s', mask.shape)
	logging.info('shape of label: %s', label.shape)

	from mxnet.gluon.data import ArrayDataset, DataLoader
	return DataLoader(
		ArrayDataset(feature, data, label, mask),
		shuffle		= shuffle,
		batch_size	= training['batch_size'],
		num_workers	= 4,
		last_batch	= 'rollover',
	) 

# def dataloader_all_sensors_seq2seq(args, setting):
#     	train, eval, test = dataloader(setting['dataset'], args)
# 	scaler = utils.Scaler(train, args.region)
# 	return dataiter_all_sensors_seq2seq(train, scaler, setting, args), \
# 		   dataiter_all_sensors_seq2seq(eval, scaler, setting, args, shuffle=False), \
# 		   dataiter_all_sensors_seq2seq(test, scaler, setting, args, shuffle=False), \
# 		   scaler

def dataloader_all_sensors_seq2seq(args, setting):
	# train, eval, test = dataloader(setting['dataset'], args)
	scaler = utils.Scaler(args['train']['Y'], '03-07')
	return dataiter_all_sensors_seq2seq('train', scaler, setting, args), \
		   dataiter_all_sensors_seq2seq('val', scaler, setting, args, shuffle=False), \
		   dataiter_all_sensors_seq2seq('test', scaler, setting, args, shuffle=False), \
		   scaler

def dataiter_all_sensors_seq2seq(setname, scaler, setting, args, shuffle=True):
	dataset = setting['dataset']
	training = setting['training']

	mdata = args
	NUM_NODES = len(mdata['grid_list'])
	XC = scaler.transform(mdata[setname]['XC'])
	TEC = np.tile(np.expand_dims(mdata[setname]['TEC'], 2), [1, 1, NUM_NODES, 1])

	Y = np.expand_dims(mdata[setname]['Y'], 1)
	TEY = np.tile(np.expand_dims(np.expand_dims(mdata[setname]['TEY'], 1), 2), [1, 1, NUM_NODES, 1])

	# df_fill = utils.fill_missing(df)
	# df_fill = scaler.transform(df)
	# NUM_NODES = df.shape[1]
	# print('NUM_NODES:', NUM_NODES)

	# n_timestamp = df_fill.shape[0]
	# data_list = [np.expand_dims(df_fill.values, axis=-1)]

	# # time in day
	# time_idx = (df_fill.index.values - df_fill.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
	# time_in_day = np.tile(time_idx, [1, NUM_NODES, 1]).transpose((2, 1, 0))
	# data_list.append(time_in_day)

	# # day in week
	# time_ind = df.index.weekday
	# dayofweek = np.tile(time_ind, [1, NUM_NODES, 1]).transpose((2, 1, 0))
	# data_list.append(dayofweek)
	
	# # temporal feature
	# temporal_feature = np.concatenate(data_list, axis=-1)

	# print('temporal_feature: ', temporal_feature.shape)


	geo_feature, _ = get_geo_feature(dataset, args)

	input_len = dataset['input_len']
	output_len = dataset['output_len']
	feature, data, mask, label  = [], [], [], []

	data = np.concatenate((XC, TEC), -1)
	label = np.concatenate((Y, TEY), -1)
	mask= np.ones(label[..., :1].shape, np.float32)
	feature = [geo_feature for _ in range(data.shape[0])]

	# for i in range(len(data.shape[0])):
    		
	# 	data.append(temporal_feature[i: i + input_len])

	# 	_mask = np.array(df.iloc[i + input_len: i + input_len + output_len] > 1e-5, dtype=np.float32)
	# 	mask.append(_mask)

	# 	label.append(temporal_feature[i + input_len: i + input_len + output_len])
		
	# 	feature.append(geo_feature)

	# 	if i % 1000 == 0:
	# 		logging.info('Processing %d timestamps', i)

	data = mx.nd.array(np.stack(data))
	label = mx.nd.array(np.stack(label))
	# mask = mx.nd.array(np.expand_dims(np.stack(mask), axis=3))
	mask = mx.nd.array(np.stack(mask))
	feature = mx.nd.array(np.stack(feature))

	logging.info('shape of feature: %s', feature.shape)
	logging.info('shape of data: %s', data.shape)
	logging.info('shape of mask: %s', mask.shape)
	logging.info('shape of label: %s', label.shape)

	from mxnet.gluon.data import ArrayDataset, DataLoader
	if setname != 'test':
		return DataLoader(
			ArrayDataset(feature, data, label, mask),
			shuffle		= shuffle,
			batch_size	= training['batch_size'],
			num_workers	= 4,
			last_batch	= 'rollover',
		) 
	else:
		return DataLoader(
			ArrayDataset(feature, data, label, mask),
			shuffle		= shuffle,
			batch_size	= 10,
			num_workers	= 4,
			last_batch	= 'rollover',
		) 

def seq2instance(data, P, Q):
    print(data.shape)
    num_step = data.shape[0]
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, *data.shape[1:]))
    y = np.zeros(shape = (num_sample, Q, *data.shape[1:]))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y


def row_normalize(a):
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums[:, np.newaxis]
    return new_matrix
