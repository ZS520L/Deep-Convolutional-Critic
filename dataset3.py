import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- 步骤 1: 复用之前编写的、适用于CWRU格式的核心加载函数 ---
def load_cwru_signal(file_path: str) -> np.ndarray:
    """
    从CWRU格式的.mat文件中加载驱动端(DE)的振动信号。
    该函数会自动查找以 "_DE_time" 结尾的键。
    """
    try:
        mat_contents = sio.loadmat(file_path)
        de_signal_key = None
        for key in mat_contents.keys():
            if key.endswith("_DE_time"):
                de_signal_key = key
                break
        
        if de_signal_key:
            return mat_contents[de_signal_key].flatten()
        else:
            # 如果找不到DE_time，尝试找FE_time作为备用
            fe_signal_key = None
            for key in mat_contents.keys():
                if key.endswith("_FE_time"):
                    fe_signal_key = key
                    break
            if fe_signal_key:
                 return mat_contents[fe_signal_key].flatten()
            return np.array([]) # 如果两者都找不到，则失败

    except (FileNotFoundError, ValueError) as e:
        # ValueError可能在.mat文件损坏时发生
        return np.array([])

# --- 步骤 2: 构建 PyTorch Dataset 类 ---
class CWRUDataset(Dataset):
    """
    用于CWRU格式轴承故障数据集的PyTorch Dataset类。
    
    该类会自动处理以下任务：
    1. 遍历文件夹，读取.mat文件。
    2. 根据文件名（是否包含'normal'）分配标签（0为正常，1为异常）。
    3. 使用滑动窗口将信号切分为多个样本段。
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
            # 1. 根据文件名确定标签
            # 使用 .lower() 使其不区分大小写，更稳健
            label = 0 if 'normal' in file_path.name.lower() else 1
            
            # 2. 加载信号
            signal = load_cwru_signal(str(file_path))
            
            if signal.size < self.window_size:
                continue # 如果信号太短，跳过

            # 3. 滑动窗口切片
            num_segments = (len(signal) - self.window_size) // self.step_size + 1
            for i in range(num_segments):
                start_idx = i * self.step_size
                end_idx = start_idx + self.window_size
                segment = signal[start_idx:end_idx]
                
                # <--- 修改这里：根据参数决定是否执行归一化 ---
                if self.use_normalization:
                    # Z-score标准化 (对每个样本段独立进行)
                    mean = np.mean(segment)
                    std = np.std(segment)
                    if std > 1e-6: # 避免除以零
                        segment = (segment - mean) / std
                
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

# --- 步骤 3: 使用示例 ---
if __name__ == '__main__':
    # 请将此路径修改为您电脑上包含CWRU格式.mat文件的文件夹路径
    # 例如 'E:\CWRU\12k_Drive_End_Bearing_Fault_Data'
    DATA_DIR = r'D:\Pro\demo\guobiao\CWRU\raw'
    
    print("--- 1. 创建数据集实例 ---")
    try:
        cwru_dataset = CWRUDataset(root_dir=DATA_DIR, window_size=2048, step_size=1024)
        
        print(f"\n数据集总样本数: {len(cwru_dataset)}")
        
        if len(cwru_dataset) > 0:
            first_sample, first_label = cwru_dataset[0]
            print(f"第一个样本的形状: {first_sample.shape}")
            print(f"第一个样本的标签: {first_label.item()}")

        print("\n--- 2. 使用 DataLoader 加载数据 ---")
        data_loader = DataLoader(
            dataset=cwru_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=0
        )
        
        data_batch, labels_batch = next(iter(data_loader))
        print(f"\n从DataLoader获取的一个批次数据:")
        print(f"数据批次的形状: {data_batch.shape}")
        print(f"标签批次的形状: {labels_batch.shape}")

    except (FileNotFoundError, StopIteration) as e:
        print(f"\n发生错误: {e}")
        print("请确认您的DATA_DIR路径设置是否正确，并且该文件夹下包含有效的.mat文件。")