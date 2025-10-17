# -*- encoding: utf-8 -*-
'''
file       : deep.py
Date       : 2025/02/11 20:06:22
Email      : qiang.wang@stu.xidian.edu.cn
Author     : qwangxdu
'''

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
from datasets.base import SignalDataset

class DEAPDataset(SignalDataset):
    """
    DEAP 数据集加载类，提供对 DEAP 数据集的特征加载与处理功能，支持按通道、标签和受试者筛选数据。
    
    数据集目录结构要求：
    - root_path/
        - s01.dat
        - s02.dat
        - ...
        - s32.dat
        - label.mat

    参数：
        root_path (str): 数据集根路径 (默认: ".\\deap")
        channels (List[str]): 选择的EEG通道列表，None表示全选 (默认: None)
        labels (List[str]): 选择的标签 (默认: None)
        subjects (List[int]): 选择的受试者ID列表，None表示全选 (默认: None)
        window_sec (int): 每个信号片段的时间窗口大小（秒）（默认: 1）
    """

    # 定义 DEAP 数据集 EEG 的通道列表
    CHANNELS_LIST = [
        'FP1', 'AF3', 'F3',  'F7',  'FC5', 'FC1', 'C3',  'T7',  'CP5', 'CP1',
        'P3',  'P7',  'PO3', 'O1',  'Oz',  'Pz',  'FP2', 'AF4', 'Fz',  'F4',
        'F8',  'FC6', 'FC2', 'Cz',  'C4',  'T8',  'CP6', 'CP2', 'P4',  'P8',
        'PO4', 'O2',  #'hEOG','vEOG','zEMG','tEMG','GSR', 'Resp','Plet','Temp'
    ]
    
    # 定义 DEAP 数据集的标签列表
    LABELS_LIST = ['valence', 'arousal', 'dominance', 'liking']
    EEG_SAMPLING_RATE = 128  # 采样率，单位为Hz
    BASELINE_DURATION = 3  # 基线持续时间，单位为秒
    STIMULUS_DURATION = 60  # 刺激持续时间，单位为秒

    def __init__(
        self,
        root_path: str = ".\\deap",
        channels: List[str] = None,
        labels: Union[str, List[str]] = None,
        window_sec: int = 1,
        step_sec: Optional[float] = None, 
        feature_name: str = None,
        **kwargs,
    ):
        # 初始化参数
        super(DEAPDataset, self).__init__(
            root_path=root_path,
            channels=channels,
            labels=labels,
            window_sec=window_sec,
            step_sec=step_sec,
            feature_name=feature_name,
            **kwargs
        )
        self.num_of_subjects = 32

        # 缓存基线和刺激数据
        self._baseline_cache, self._stimulus_cache = \
            self._process_all_subjects()

    def get_baseline(self):
        return self._baseline_cache
    
    def get_stimulus(self):
        return self._stimulus_cache
    
    def get_dataset(self):
        changed_data = self._subtract_baseline(
            self._stimulus_cache["data"], self._stimulus_cache["groups"],  
            self._baseline_cache["data"], self._baseline_cache["groups"])
        # changed_data = self._stimulus_cache["data"]
        return {
            "data" : changed_data,
            "labels" : self._stimulus_cache["labels"],
            "groups" : self._stimulus_cache["groups"]
        }

    def get_feature_dim(self):
        return self._stimulus_cache["data"].shape[-1]

    
    def _process_all_subjects(self):
        baseline_data_list = []  # List to hold all baseline feature data
        baseline_group_list = []  # List to hold all baseline group data
        stimulus_data_list = []  # List to hold all stimulus feature data
        stimulus_group_list = []  # List to hold all stimulus group data
        stimulus_labels_list = []  # List to hold all stimulus labels data
        
        for info in self._get_meta_info():
            # Read data for each subject
            samples, labels = self._load_file(info["file_path"])

            # Process the data for this subject
            baseline, stimulus = self._process_one_subject(info["subject"], samples, labels)

            # Append the data to the respective lists
            baseline_data_list.append(baseline["data"])
            baseline_group_list.append(baseline["groups"])

            stimulus_data_list.append(stimulus["data"])
            stimulus_group_list.append(stimulus["groups"])
            stimulus_labels_list.append(stimulus["labels"])

        # Once all subjects are processed, convert lists to numpy arrays
        _baseline_cache = {
            "data": np.vstack(baseline_data_list),  # Stack all baseline feature data
            "groups": np.vstack(baseline_group_list)  # Stack all baseline group data
        }
        _stimulus_cache = {
            "data": np.vstack(stimulus_data_list),  # Stack all stimulus feature data
            "groups": np.vstack(stimulus_group_list),  # Stack all stimulus group data
            "labels": np.vstack(stimulus_labels_list)  # Concatenate all stimulus labels
        }
        return _baseline_cache, _stimulus_cache

    def _process_one_subject(
        self,
        subject_id: int,
        samples: np.ndarray,
        labels: np.ndarray,
    ):
        # 先根据通道和标签筛选数据
        samples = samples[:, self.channel_indices, :]
        labels = labels[:, self.label_indices]

        # 提取基线和刺激数据
        baseline_data = self._extract_time_window(samples, start=0, duration=self.BASELINE_DURATION)
        stimulus_data = self._extract_time_window(samples, start=self.BASELINE_DURATION, duration=self.STIMULUS_DURATION)

        # 分割基线和刺激数据
        baseline_group, baseline_data, _ = self._segment_signal(baseline_data, None)
        stimulus_group, stimulus_data, stimulus_labels = self._segment_signal(stimulus_data, labels)
        # 提取特征
        baseline_feature = self.feature_extractor(baseline_data)
        stimulus_feature = self.feature_extractor(stimulus_data)

        # 填充受试者group信息
        baseline_group = np.column_stack((np.full_like(baseline_group, subject_id), baseline_group))
        stimulus_group = np.column_stack((np.full_like(stimulus_group, subject_id), stimulus_group))

        # 构建基线和刺激的字典
        baseline = {
            "data": baseline_feature,
            "groups": baseline_group
        }

        stimulus = {
            "data": stimulus_feature,
            "groups": stimulus_group,
            "labels": stimulus_labels
        }

        return baseline, stimulus

    @staticmethod
    def _load_file(file_path: Path):
        """加载文件数据"""
        with open(file_path, 'rb') as f:
            pkl = pickle.load(f, encoding='latin1')
        
        samples = pkl["data"]
        labels = pkl["labels"]

        return samples, labels


    def _get_meta_info(self) -> List[Dict]:
        """获取数据集的元信息，包括受试者和文件路径"""
        meta_info = []
        for subject in list(range(1, self.num_of_subjects + 1)):
            file_path = self.root_path / f"s{subject:02d}.dat"
            if file_path.exists():
                meta_info.append({
                    "subject": subject,
                    "file_path": file_path
                })
        if not meta_info:
            raise FileNotFoundError("No valid subject data files found in the specified root path.")
        
        return meta_info
    
    
    