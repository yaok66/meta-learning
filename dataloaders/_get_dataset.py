# -*- encoding: utf-8 -*-
'''
file       :_get_dataset.py
Date       :2025/02/17 20:56:28
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datasets import SEEDFeatureDataset, SEEDIVFeatureDataset, DEAPDataset, DreamerDataset

def get_dataset(args):
    # tmp_path = f"E:\\EEG_DataSets\\Preprocessed\\"
    # tmp_data_path = tmp_path + f"{args.dataset_name}_{args.session}_{args.feature_name}_{args.emotion}_{args.window_sec}s_{args.step_sec}s.npz"

    # if os.path.exists(tmp_data_path):
    #     temp = np.load(tmp_data_path)
    #     data, one_hot_mat, Group = temp['data'], temp['one_hot_mat'], temp['groups']
    #     setattr(args, "feature_dim", data.shape[-1])
    #     setattr(args, "num_of_class", one_hot_mat.shape[-1])

    # else:
    if args.dataset_name == "seed3":
        data, one_hot_mat, Group = get_seed(args)
    elif args.dataset_name == "seed4":
        data, one_hot_mat, Group = get_seediv(args)
    elif args.dataset_name == "deap":
        data, one_hot_mat, Group = get_deap(args)
    elif args.dataset_name == "dreamer":
        data, one_hot_mat, Group = get_dreamer(args)
    # np.savez(tmp_data_path, data=data, one_hot_mat=one_hot_mat, groups=Group)
    return {
        "data": data,
        "one_hot_mat": one_hot_mat, 
        "groups": Group
    }


def get_seed(args):

    num_of_class = 3
    num_of_channels = 62

    SEED = SEEDFeatureDataset(args.seed3_path, sessions=args.session)
    feature_dim = num_of_channels * SEED.get_feature_dim()
    dataset = SEED.get_dataset()

    data, Label, Group = dataset["data"], dataset["labels"], dataset["groups"]
    Label += 1 # begin from 0
    data = data.reshape(-1, feature_dim)
    subject_ids = Group[:, 0] # subject ID 

    setattr(args, "feature_dim", feature_dim)
    setattr(args, "num_of_class", num_of_class)

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in np.unique(subject_ids):
        data[subject_ids==i] = min_max_scaler.fit_transform(data[subject_ids == i])

    one_hot_mat = np.eye(len(Label), num_of_class)[Label].astype("float32")
    return data, one_hot_mat, Group

def get_seediv(args):
    num_of_class = 4
    num_of_channels = 62

    SEEDIV = SEEDIVFeatureDataset(args.seed4_path, sessions=args.session)
    feature_dim = num_of_channels * SEEDIV.get_feature_dim()
    dataset = SEEDIV.get_dataset()

    data, Label, Group = dataset["data"], dataset["labels"], dataset["groups"]
    data = data.reshape(-1, feature_dim)
    subject_ids = Group[:, 0] # subject ID 

    setattr(args, "feature_dim", feature_dim)
    setattr(args, "num_of_class", num_of_class)

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in np.unique(subject_ids):
        data[subject_ids==i] = min_max_scaler.fit_transform(data[subject_ids == i])

    one_hot_mat = np.eye(len(Label), num_of_class)[Label].astype("float32")
    return data, one_hot_mat, Group


def avg_moving(data,  groups, window_size=10):
    normalized_data = []

    def avg_moving_func(data, window_size=10):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    for subject_id in np.unique(groups[:, 0]):
        subject_mask = groups[:, 0] == subject_id
        subject_data = data[subject_mask]
        subject_trials = groups[subject_mask][:, 1]
        unique_trials = np.unique(subject_trials)
        for trial in unique_trials:
            trial_mask = subject_trials == trial
            trial_data = subject_data[trial_mask]
            trial_data_normalized = np.apply_along_axis(avg_moving_func, 0, trial_data)
            
            normalized_data.append(trial_data_normalized)
    data = np.vstack(normalized_data)
    return data

def get_deap(args):
    num_of_class = 2
    num_of_channels = 32

    params = {
        "feature_name" : args.feature_name,
        "window_sec" : args.window_sec, 
        "step_sec" : args.step_sec,
        "labels" : args.emotion
    }

    DEAP = DEAPDataset(args.deap_path, **params)
    feature_dim = num_of_channels * DEAP.get_feature_dim()
    dataset = DEAP.get_dataset()

    data, Label, Group = dataset["data"], dataset["labels"], dataset["groups"]
    data = data.reshape(-1, feature_dim)
    data = avg_moving(data, Group)
    Label = Label.reshape(-1)
    setattr(args, "feature_dim", feature_dim)
    setattr(args, "num_of_class", num_of_class)
    Label = (Label > 5).astype(int)
    one_hot_mat = np.eye(len(Label), num_of_class)[Label].astype("float32")
    return data, one_hot_mat, Group

def get_dreamer(args):
    num_of_class = 2
    num_of_channels = 14
    params = {
        "feature_name" : args.feature_name,
        "window_sec" : args.window_sec, 
        "step_sec" : args.step_sec,
        "labels" : args.emotion
    }

    DREAMER = DreamerDataset(args.dreamer_path, **params)
    feature_dim = num_of_channels * DREAMER.get_feature_dim()
    dataset = DREAMER.get_dataset()

    data, Label, Group = dataset["data"], dataset["labels"], dataset["groups"]
    data = data.reshape(-1, feature_dim)
    Label = Label.reshape(-1)

    setattr(args, "feature_dim", feature_dim)
    setattr(args, "num_of_class", num_of_class)

    Label = (Label > 3).astype(int)

    one_hot_mat = np.eye(len(Label), num_of_class)[Label].astype("float32")

    return data, one_hot_mat, Group