import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- 步骤 1: 核心数据加载函数 (无变化) ---
def load_mfpt_signal(file_path: str) -> np.ndarray:
    """
    从MFPT数据集的.mat文件中加载振动信号数据。
    """
    try:
        mat_contents = sio.loadmat(file_path)
        signal_vector = mat_contents['bearing']['gs'][0, 0].flatten()
        return signal_vector
    except (FileNotFoundError, KeyError) as e:
        return np.array([])

# --- 步骤 2: 构建 PyTorch Dataset 类 (已修改) ---
class MFPTDataset(Dataset):
    """
    用于MFPT轴承故障数据集的PyTorch Dataset类。
    
    该类会自动处理以下任务：
    1. 遍历文件夹，读取.mat文件。
    2. 根据文件名（是否包含'baseline'）分配标签（0为正常，1为异常）。
    3. 使用滑动窗口将信号切分为多个样本段，并进行归一化。
    """
    def __init__(self, root_dir: str, window_size: int = 1024, step_size: int = 512, use_normalization: bool = True):
        """
        初始化数据集。
        
        参数:
            root_dir (str): 包含.mat文件的数据集根目录。
            window_size (int): 每个样本段的长度（窗口大小）。默认为1024。
            step_size (int): 滑动窗口的步长。默认为512。
        """
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.step_size = step_size
        
        # <--- 新增这一行：保存归一化设置 ---
        self.use_normalization = use_normalization
        
        self.samples = []
        self.labels = []
        
        self._prepare_data()

    def _prepare_data(self):
        """
        遍历文件、加载数据、切片并填充 self.samples 和 self.labels 列表。
        """
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"错误：提供的路径不是一个有效的目录: {self.root_dir}")

        mat_files = sorted(list(self.root_dir.glob('*.mat')))
        print(f"在 '{self.root_dir}' 中找到 {len(mat_files)} 个 .mat 文件。正在处理...")
        
        for file_path in tqdm(mat_files, desc="处理数据文件"):
            label = 0 if 'baseline' in file_path.name.lower() else 1
            signal = load_mfpt_signal(str(file_path))
            
            if signal.size < self.window_size:
                continue

            num_segments = (len(signal) - self.window_size) // self.step_size + 1
            for i in range(num_segments):
                start_idx = i * self.step_size
                end_idx = start_idx + self.window_size
                segment = signal[start_idx:end_idx]
                
                # =====================================================
                # === 新增的归一化操作 ===
                # =====================================================
                # <--- 修改这里：根据参数决定是否执行归一化 ---
                if self.use_normalization:
                    # Z-score标准化 (对每个样本段独立进行)
                    mean = np.mean(segment)
                    std = np.std(segment)
                    if std > 1e-6: # 避免除以零
                        segment = (segment - mean) / std
                
                # =====================================================
                # === 归一化操作结束 ===
                # =====================================================
                
                self.samples.append(segment)
                self.labels.append(label)
        
        print(f"数据处理完成！共生成 {len(self.samples)} 个样本段。")

    def __len__(self) -> int:
        """返回数据集中样本的总数。"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据索引获取一个样本。
        """
        segment_tensor = torch.from_numpy(self.samples[idx]).float().unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return segment_tensor, label_tensor

# --- 步骤 3: 使用示例 (无变化) ---
if __name__ == '__main__':
    # 请将此路径修改为您电脑上包含MFPT .mat文件的文件夹路径
    DATA_DIR = r'E:\AI\MFPT-Fault-Data-Sets-20200227T131140Z-001\MFPT Fault Data Sets\MFPT'
    
    print("--- 1. 创建数据集实例 ---")
    mfpt_dataset = MFPTDataset(root_dir=DATA_DIR)
    
    print(f"\n数据集总样本数: {len(mfpt_dataset)}")
    
    if len(mfpt_dataset) > 0:
        first_sample, first_label = mfpt_dataset[0]
        print(f"第一个样本的形状: {first_sample.shape}")
        print(f"第一个样本的数据类型: {first_sample.dtype}")
        print(f"第一个样本的标签: {first_label}")
        print(f"第一个样本的标签类型: {first_label.dtype}")

    print("\n--- 2. 使用 DataLoader 加载数据 ---")
    data_loader = DataLoader(
        dataset=mfpt_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
    try:
        data_batch, labels_batch = next(iter(data_loader))
        print(f"\n从DataLoader获取的一个批次数据:")
        print(f"数据批次的形状: {data_batch.shape}")
        print(f"标签批次的形状: {labels_batch.shape}")
    except StopIteration:
        print("\n数据集为空或太小，无法创建一个完整的批次。")
