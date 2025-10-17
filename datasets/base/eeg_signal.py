# -*- encoding: utf-8 -*-
'''
file       :eeg_signal.py
Date       :2025/02/13 12:06:28
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Optional
from datasets.base import FeatureExtractor

class SignalDataset(ABC):
    """EEG 信号处理的抽象类
    """
    CHANNELS_LIST: List[str] = []
    LABELS_LIST: List[str] = []
    EEG_SAMPLING_RATE: int = 128

    # # for DEAP Datasets
    # BASELINE_DURATION: int = 0
    # STIMULUS_DURATION: int = 0

    def __init__(
        self,
        root_path: str,
        channels: List[str] = None,
        labels: Union[str, List[str]] = None,
        window_sec: int = 1,
        step_sec: Optional[float] = None,
        feature_name: str = None,
        **kwargs
    ):
        super(SignalDataset, self).__init__()
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            raise FileNotFoundError(f"The specified root path does not exist: {self.root_path}")
        self.channel_indices = self._get_channel_indices(channels)
        self.label_indices = self._get_label_indices(labels)
        self.window_sec = window_sec
        self.step_sec = step_sec if step_sec else window_sec
        self.feature_extractor = FeatureExtractor(feature_name, **kwargs)

    @abstractmethod
    def _get_meta_info(self) -> List[Dict]:
        """获取数据集元信息 (子类必须实现)"""
        pass

    @abstractmethod
    def _load_file(self, file_path: Path ):
        """加载文件数据 (子类必须实现)"""
        pass

    @abstractmethod
    def _process_all_subjects(self):
        pass

    def _segment_signal(
        self, 
        signal: np.ndarray, 
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ...]:
        """信号分窗处理"""
        if signal.ndim != 3:
            raise ValueError("Signal must be a 3D array with shape (trials, channels, points)")
        if labels is not None and labels.ndim != 2:
            raise ValueError("Labels must be a 2D array with shape (trials, labels)")
        window_points = self.window_sec * self.EEG_SAMPLING_RATE
        step_points = int(self.step_sec * self.EEG_SAMPLING_RATE)
        n_points = signal.shape[-1]

        # 对于DEAP基线等，可能存在窗口大小大于基线大小
        if n_points < window_points:
            window_points = n_points

        # 计算切片数量
        n_slices = (n_points - window_points) // step_points + 1
        slices = np.stack([
            signal[..., i*step_points : i*step_points+window_points] 
            for i in range(n_slices)
        ], axis=1) # 添加切片维度，放在channels前面

        # 重组维度
        n_trials, n_slices, n_channels, _ = slices.shape
        segments = slices.reshape(-1, n_channels, window_points)
        groups = np.repeat(np.arange(1, n_trials+1), n_slices)

        # 处理标签
        segmented_labels = None
        if labels is not None:
            segmented_labels = np.repeat(labels, n_slices, axis=0)

        return groups, segments, segmented_labels
    
    def _extract_time_window(self, signal: np.ndarray, start: int, duration: int) -> np.ndarray:
        """提取指定时间窗口"""
        start_idx = start * self.EEG_SAMPLING_RATE
        end_idx = (start + duration) * self.EEG_SAMPLING_RATE
        return signal[..., start_idx:end_idx]
    
    @staticmethod
    def _subtract_baseline(
        stimulus_data: np.ndarray, 
        stim_groups: np.ndarray,
        baseline_data: np.ndarray, 
        base_groups: np.ndarray
    ) -> np.ndarray:
        """基线校正"""
        if stim_groups.ndim != 2 or base_groups.ndim != 2:
            raise ValueError("Groups must be 2D arrays with shape (subjects, trials)")
        corrected = np.zeros_like(stimulus_data)
        unique_subjects = np.unique(base_groups[:, 0])
        for subject in unique_subjects:
            subject_mask = base_groups[:, 0] == subject
            unique_trials = np.unique(base_groups[subject_mask, 1])
            for trial in unique_trials:
                trial_mask = (base_groups[:, 0] == subject) & (base_groups[:, 1] == trial)
                base_mean = baseline_data[trial_mask].mean(axis=0)
                stim_mask = (stim_groups[:, 0] == subject) & (stim_groups[:, 1] == trial)
                corrected[stim_mask] = stimulus_data[stim_mask] - base_mean
        return corrected

    def _get_channel_indices(self, channels: List[str]) -> np.ndarray:
        """获取通道索引"""
        if not channels:
            return np.arange(len(self.CHANNELS_LIST))
        return np.where(np.isin(self.CHANNELS_LIST, channels))[0]

    def _get_label_indices(self, labels: List[str]) -> np.ndarray:
        """获取标签索引"""
        if not labels:
            return np.arange(len(self.LABELS_LIST))
        return np.where(np.isin(self.LABELS_LIST, labels))[0]

