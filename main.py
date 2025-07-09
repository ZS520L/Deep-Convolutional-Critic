import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import traceback

# --- 0. 基础设置 ---
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
torch.manual_seed(42); np.random.seed(42); random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 数据集类导入 ---
# (假设这些类已正确定义在同级目录下)
try:
    from dataset1 import MFPTDataset
    from dataset2 import CWRUDataset as SEUDataset
    from dataset3 import CWRUDataset as CWRUDataset
except ImportError:
    print("错误：无法导入一个或多个数据集类。请确保 dataset1.py, dataset2.py, dataset3.py 文件存在。")
    # 创建虚拟类以允许代码在没有数据集文件的情况下至少可以解析
    class BaseDataset(Dataset):
        def __init__(self, *args, **kwargs): self.samples, self.labels = [], []
        def __len__(self): return 0
        def __getitem__(self, idx): raise IndexError
    MFPTDataset = SEUDataset = CWRUDataset = BaseDataset
    print("已创建虚拟数据集类以继续运行。")


# --- 2. 全局配置 ---
DATASET_CONFIGS = {
    'CWRU': {'class': CWRUDataset, 'path': r'E:\AI\CWRU-dataset-main\ALL'},
    'SEU': {'class': SEUDataset, 'path': r'E:\AI\Mechanical-datasets-master\dataset'},
    'MFPT': {'class': MFPTDataset, 'path': r'E:\AI\MFPT-Fault-Data-Sets-20200227T131140Z-001\MFPT Fault Data Sets\MFPT'}
}
PARAMS = {'window_size': 1024, 'step_size': 512, 'batch_size': 128, 'lr': 0.001,
          'epochs': 50, 'train_normal_split': 0.5, 'use_normalization': True,
          'latent_dim': 100, 'lambda_gp': 10}

# ✅ 新增: 噪声水平配置
NOISE_LEVELS_TO_TEST = [0.0, 0.1, 0.2, 0.5, 1.0] # 0.0代表无噪声, 其他值是噪声标准差与信号标准差的比率

# --- 3. 模型定义 ---
# (所有模型定义保持不变)
class DCC(nn.Module):
    def __init__(self, signal_length=1024):
        super(DCC, self).__init__()
        self.features = nn.Sequential(
            spectral_norm(nn.Conv1d(1, 16, 128, 2, 63)), nn.LeakyReLU(0.2, inplace=True), nn.MaxPool1d(4, 4))
        with torch.no_grad():
            fc_in = self.features(torch.randn(1, 1, signal_length)).flatten().shape[0]
        self.classifier = nn.Sequential(spectral_norm(nn.Linear(fc_in, 1)))
    def forward(self, x): return self.classifier(self.features(x).view(x.size(0), -1)).squeeze()
class ConvAE(nn.Module):
    def __init__(self, signal_length=1024):
        super(ConvAE, self).__init__();
        self.encoder = nn.Sequential(nn.Conv1d(1, 16, 32, 4, 14), nn.ReLU(), nn.Conv1d(16, 32, 16, 4, 6), nn.ReLU(), nn.Conv1d(32, 64, 8, 4, 2))
        self.decoder = nn.Sequential(nn.ConvTranspose1d(64, 32, 8, 4, 2), nn.ReLU(), nn.ConvTranspose1d(32, 16, 16, 4, 6), nn.ReLU(), nn.ConvTranspose1d(16, 1, 32, 4, 14))
    def forward(self, x): return self.decoder(self.encoder(x))
class DeepSVDD_Net(nn.Module):
    def __init__(self, signal_length=1024):
        super(DeepSVDD_Net, self).__init__();
        self.net = nn.Sequential(nn.Conv1d(1, 16, 32, 4, 14), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Conv1d(16, 32, 16, 4, 6), nn.ReLU(), nn.MaxPool1d(2, 2))
        with torch.no_grad(): fc_in = self.net(torch.randn(1, 1, signal_length)).flatten().shape[0]
        self.fc = nn.Linear(fc_in, 128)
    def forward(self, x): return self.fc(self.net(x).view(x.size(0), -1))
class VAE(nn.Module):
    def __init__(self, signal_length, latent_dim):
        super(VAE, self).__init__();
        self.encoder = nn.Sequential(nn.Conv1d(1, 16, 32, 4, 14), nn.ReLU(), nn.Conv1d(16, 32, 16, 4, 6))
        with torch.no_grad(): fc_in = self.encoder(torch.randn(1, 1, signal_length)).flatten().shape[0]
        self.fc_mu = nn.Linear(fc_in, latent_dim); self.fc_logvar = nn.Linear(fc_in, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, fc_in)
        self.decoder = nn.Sequential(nn.ConvTranspose1d(32, 16, 16, 4, 6), nn.ReLU(), nn.ConvTranspose1d(16, 1, 32, 4, 14))
        self.encoder_out_shape = (-1, 32, 64)
    def encode(self, x): h = self.encoder(x).view(x.size(0), -1); return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar): std = torch.exp(0.5 * logvar); return mu + torch.randn_like(std) * std
    def decode(self, z): h = self.decoder_fc(z).view(self.encoder_out_shape); return self.decoder(h)
    def forward(self, x): mu, logvar = self.encode(x); z = self.reparameterize(mu, logvar); return self.decode(z), mu, logvar
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, n_layers=2, output_dim=512):
        super(LSTMPredictor, self).__init__();
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x): lstm_out, _ = self.lstm(x); return self.fc(lstm_out[:, -1, :])
class DC_Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(DC_Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 512, 4, 1, 0, bias=False), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.ConvTranspose1d(512, 256, 4, 4, 0, bias=False), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.ConvTranspose1d(256, 128, 4, 4, 0, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, 4, 4, 0, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.ConvTranspose1d(64, 1, 4, 4, 0, bias=False), nn.Tanh()
        )
    def forward(self, input):
        return self.main(input.view(input.size(0), input.size(1), 1))
class DC_Discriminator(nn.Module):
    def __init__(self, is_wgan=False):
        super(DC_Discriminator, self).__init__()
        self.is_wgan = is_wgan
        self.main = nn.Sequential(
            nn.Conv1d(1, 64, 4, 4, 0, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 4, 4, 0, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, 4, 4, 0, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, 4, 4, 0, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 1, 4, 1, 0, bias=False)
        )
    def forward(self, input):
        output = self.main(input).view(-1, 1).squeeze(1)
        return output if self.is_wgan else torch.sigmoid(output)

# --- 4. 训练和评估函数 ---

# ✅ 新增: 噪声注入函数
def add_gaussian_noise(data, noise_level):
    """
    向数据中添加高斯噪声。
    :param data: numpy 数组形式的原始数据
    :param noise_level: 噪声水平，作为原始数据标准差的乘数
    :return: 添加了噪声的数据
    """
    if noise_level == 0.0:
        return data
    sigma = noise_level * np.std(data)
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise

# ✅ 修改: get_dataloaders 函数，增加 noise_level 参数
def get_dataloaders(dataset_name, config, params, noise_level=0.0):
    print(f"\n--- [数据准备 for {dataset_name}, Noise Level: {noise_level}] ---")
    DatasetClass = config['class']
    full_dataset = DatasetClass(root_dir=config['path'], window_size=params['window_size'], step_size=params['step_size'], use_normalization=params['use_normalization'])
    if not full_dataset or len(full_dataset) == 0: return (None,) * 5
    
    normal_indices = [i for i, l in enumerate(full_dataset.labels) if l == 0]
    anomaly_indices = [i for i, l in enumerate(full_dataset.labels) if l == 1]
    if not normal_indices: 
        print(f"警告: 在 {dataset_name} 数据集中未找到正常样本。")
        return (None,) * 5

    # 分割正常数据用于训练和测试
    train_normal_indices, test_normal_indices = train_test_split(
        normal_indices, train_size=params['train_normal_split'], random_state=42
    )
    
    # 准备训练集 (始终是干净的)
    train_segments_np = np.array([full_dataset.samples[i] for i in train_normal_indices])
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_segments_np).float().unsqueeze(1)), 
                              batch_size=params['batch_size'], shuffle=True, drop_last=True)
    
    # 准备测试集
    test_normal_segments = [full_dataset.samples[i] for i in test_normal_indices]
    test_anomaly_segments = [full_dataset.samples[i] for i in anomaly_indices]
    test_segments_np = np.array(test_normal_segments + test_anomaly_segments)
    y_test_np = np.array([0] * len(test_normal_segments) + [1] * len(test_anomaly_segments))

    # ✅ 对测试集注入噪声
    if noise_level > 0.0:
        print(f"  -> 对测试集注入噪声 (Level: {noise_level})...")
        test_segments_np = add_gaussian_noise(test_segments_np, noise_level)

    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_segments_np).float().unsqueeze(1), 
                                           torch.from_numpy(y_test_np).long()), 
                             batch_size=params['batch_size'])
    
    print(f"总样本数: {len(full_dataset)} (正常: {len(normal_indices)}, 异常: {len(anomaly_indices)})")
    print(f"训练集大小 (正常样本): {len(train_segments_np)}")
    print(f"测试集大小: {len(test_segments_np)} (正常: {len(test_normal_segments)}, 异常: {len(test_anomaly_segments)})")

    return train_loader, test_loader, train_segments_np, test_segments_np, y_test_np

# (所有模型的 train_evaluate_* 函数保持不变，因为噪声已在数据加载层处理)
def train_evaluate_dcc(train_loader, test_loader, y_test_np, params):
    model = DCC(signal_length=params['window_size']).to(device); optimizer = optim.Adam(model.parameters(), lr=params['lr']); model.train()
    for _ in tqdm(range(params['epochs']), desc="训练 DCC", leave=False, ncols=80):
        for batch in train_loader: optimizer.zero_grad(); loss = -torch.mean(model(batch[0].to(device))); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): all_scores = np.concatenate([model(s.to(device)).cpu().numpy() for s, l in test_loader])
    return roc_auc_score(y_test_np, -all_scores)
def train_evaluate_convae(train_loader, test_loader, y_test_np, params):
    model = ConvAE(signal_length=params['window_size']).to(device); optimizer = optim.Adam(model.parameters(), lr=params['lr']); criterion = nn.MSELoss(); model.train()
    for _ in tqdm(range(params['epochs']), desc="训练 ConvAE", leave=False, ncols=80):
        for batch in train_loader: signals = batch[0].to(device); optimizer.zero_grad(); reconstructed = model(signals); loss = criterion(reconstructed, signals); loss.backward(); optimizer.step()
    model.eval(); all_scores = []
    with torch.no_grad():
        for signals, _ in test_loader: reconstructed = model(signals.to(device)); errors = torch.mean((signals.to(device) - reconstructed).pow(2), dim=[1, 2]); all_scores.extend(errors.cpu().numpy())
    return roc_auc_score(y_test_np, all_scores)
def train_evaluate_deepsvdd(train_loader, test_loader, y_test_np, params):
    model = DeepSVDD_Net(signal_length=params['window_size']).to(device); optimizer = optim.Adam(model.parameters(), lr=params['lr']); model.eval()
    with torch.no_grad(): all_embeddings = torch.cat([model(batch[0].to(device)) for batch in train_loader], dim=0); center = all_embeddings.mean(dim=0).detach()
    model.train()
    for _ in tqdm(range(params['epochs']), desc="训练 DeepSVDD", leave=False, ncols=80):
        for batch in train_loader: optimizer.zero_grad(); loss = torch.mean(torch.sum((model(batch[0].to(device)) - center).pow(2), dim=1)); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): all_scores = np.concatenate([torch.sum((model(s.to(device)) - center).pow(2), dim=1).cpu().numpy() for s, l in test_loader])
    return roc_auc_score(y_test_np, all_scores)
def train_evaluate_iforest(train_np, test_np, y_test_np):
    model = IsolationForest(contamination='auto', random_state=42, n_estimators=100); model.fit(train_np.reshape(len(train_np), -1))
    scores = model.decision_function(test_np.reshape(len(test_np), -1)); return roc_auc_score(y_test_np, -scores)
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x.view(-1), x.view(-1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()); return BCE + KLD
def train_evaluate_vae(train_loader, test_loader, y_test_np, params):
    model = VAE(params['window_size'], params['latent_dim']).to(device); optimizer = optim.Adam(model.parameters(), lr=params['lr']); model.train()
    for _ in tqdm(range(params['epochs']), desc="训练 VAE", leave=False, ncols=80):
        for batch in train_loader: signals = batch[0].to(device); optimizer.zero_grad(); recon_batch, mu, logvar = model(signals); loss = vae_loss_function(recon_batch, signals, mu, logvar); loss.backward(); optimizer.step()
    model.eval(); all_scores = []
    with torch.no_grad():
        for signals, _ in test_loader: recon, _, _ = model(signals.to(device)); recon_error = nn.functional.mse_loss(recon, signals.to(device), reduction='none').mean([1,2]); all_scores.extend(recon_error.cpu().numpy())
    return roc_auc_score(y_test_np, all_scores)
def train_evaluate_lstmpredictor(train_loader, test_loader, y_test_np, params):
    input_len = params['window_size'] // 2; output_len = params['window_size'] // 2
    model = LSTMPredictor(output_dim=output_len).to(device); optimizer = optim.Adam(model.parameters(), lr=params['lr']); criterion = nn.MSELoss(); model.train()
    for _ in tqdm(range(params['epochs']), desc="训练 LSTM-Seq2Seq", leave=False, ncols=80):
        for batch in train_loader:
            signals = batch[0].to(device); input_seq = signals[:, :, :input_len]; target_seq = signals[:, :, input_len:]
            lstm_input = input_seq.permute(0, 2, 1); target = target_seq.squeeze(1); optimizer.zero_grad()
            prediction = model(lstm_input); loss = criterion(prediction, target); loss.backward(); optimizer.step()
    model.eval(); all_scores = []
    with torch.no_grad():
        for signals, _ in test_loader:
            signals = signals.to(device); input_seq = signals[:, :, :input_len]; target_seq = signals[:, :, input_len:]
            lstm_input = input_seq.permute(0, 2, 1); target = target_seq.squeeze(1); prediction = model(lstm_input)
            errors = torch.mean((prediction - target).pow(2), dim=1); all_scores.extend(errors.cpu().numpy())
    return roc_auc_score(y_test_np, all_scores)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1: nn.init.normal_(m.weight.data, 1.0, 0.02); nn.init.constant_(m.bias.data, 0)
def anomaly_test_recon_error(netG, test_loader, params):
    netG.eval(); all_scores = []
    for signals, _ in tqdm(test_loader, desc="评估中", leave=False, ncols=80):
        signals = signals.to(device)
        z_opt = torch.randn(signals.size(0), params['latent_dim'], device=device, requires_grad=True)
        z_optimizer = optim.Adam([z_opt], lr=0.01)
        for _ in range(100):
            z_optimizer.zero_grad(); loss = nn.MSELoss()(netG(z_opt), signals); loss.backward(); z_optimizer.step()
        final_recon_error = torch.mean((netG(z_opt) - signals).pow(2), dim=[1,2])
        all_scores.extend(final_recon_error.detach().cpu().numpy())
    return all_scores
def train_evaluate_dcgan(train_loader, test_loader, y_test_np, params):
    netG = DC_Generator(params['latent_dim']).to(device); netG.apply(weights_init)
    netD = DC_Discriminator().to(device); netD.apply(weights_init)
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    for _ in tqdm(range(params['epochs']), desc="训练 DCGAN", leave=False, ncols=80):
        for batch in train_loader:
            real = batch[0].to(device); b_size = real.size(0); netD.zero_grad()
            label = torch.full((b_size,), 1., device=device); output = netD(real).view(-1)
            errD_real = criterion(output, label); errD_real.backward()
            fake = netG(torch.randn(b_size, params['latent_dim'], device=device))
            label.fill_(0.); output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label); errD_fake.backward(); optimizerD.step()
            netG.zero_grad(); label.fill_(1.); output = netD(fake).view(-1)
            errG = criterion(output, label); errG.backward(); optimizerG.step()
    all_scores = anomaly_test_recon_error(netG, test_loader, params)
    return roc_auc_score(y_test_np, all_scores)
def calculate_gradient_penalty(netD, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = netD(interpolates)
    fake = torch.autograd.Variable(torch.ones(d_interpolates.size()), requires_grad=False).to(device)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
def train_evaluate_wgan_gp(train_loader, test_loader, y_test_np, params):
    netG = DC_Generator(params['latent_dim']).to(device); netG.apply(weights_init)
    netD = DC_Discriminator(is_wgan=True).to(device); netD.apply(weights_init)
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.9))
    one = torch.tensor(1, dtype=torch.float, device=device); mone = one * -1
    for epoch in tqdm(range(params['epochs']), desc="训练 WGAN-GP", leave=False, ncols=80):
        for i, batch in enumerate(train_loader):
            netD.zero_grad(); real = batch[0].to(device); b_size = real.size(0)
            errD_real = netD(real).mean(); errD_real.backward(mone)
            fake = netG(torch.randn(b_size, params['latent_dim'], device=device))
            errD_fake = netD(fake.detach()).mean(); errD_fake.backward(one)
            gp_loss = params['lambda_gp'] * calculate_gradient_penalty(netD, real.data, fake.data, device); gp_loss.backward()
            optimizerD.step()
            if i % 5 == 0:
                netG.zero_grad(); fake = netG(torch.randn(b_size, params['latent_dim'], device=device))
                errG = netD(fake).mean(); errG.backward(mone); optimizerG.step()
    all_scores = anomaly_test_recon_error(netG, test_loader, params)
    return roc_auc_score(y_test_np, all_scores)

# --- 5. 主实验流程 ---
# ✅ 新增: 结果展示函数，用于显示表格和绘制曲线图
def display_noise_results(results, datasets, noise_levels):
    # 为每个数据集打印一个结果表格
    for ds_name in datasets:
        if ds_name not in results: continue
        
        print("\n\n" + "="*80)
        print(f" Dataset: {ds_name} - AUC vs. Noise Level ".center(80, "="))
        print("="*80)

        header = f"| {'Method':<18} |" + "".join([f" Noise {nl:<5} |" for nl in noise_levels])
        print(header)
        separator = f"| :{'-'*17}: |" + "".join([f" :{'-'*10}: |" for _ in noise_levels])
        print(separator)

        for model_name, model_results in results[ds_name].items():
            row = f"| {model_name:<18} |"
            for nl in noise_levels:
                score = model_results.get(nl, 'N/A')
                if isinstance(score, float):
                    row += f" {score:^10.4f} |"
                else:
                    row += f" {str(score):^10} |"
            print(row)
        print("="*80)

    # 为每个数据集绘制一张性能曲线图
    for ds_name in datasets:
        if ds_name not in results: continue
        plt.figure(figsize=(12, 8))
        plt.title(f'模型性能随噪声水平变化 ({ds_name} 数据集)', fontsize=16)
        
        for model_name, model_results in results[ds_name].items():
            # 确保结果按噪声水平排序
            sorted_results = sorted(model_results.items())
            x_noise = [r[0] for r in sorted_results]
            y_auc = [r[1] for r in sorted_results if isinstance(r[1], float)]
            
            # 只有当有有效浮点数结果时才绘制
            if len(y_auc) == len(x_noise):
                 plt.plot(x_noise, y_auc, marker='o', linestyle='-', label=model_name)

        plt.xlabel('噪声水平 (噪声标准差 / 信号标准差)', fontsize=12)
        plt.ylabel('AUC Score', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title='模型', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(noise_levels)
        plt.ylim(0.4, 1.02) # 设定合理的Y轴范围
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局为图例留出空间
        
        # 保存图表
        output_filename = f'results_{ds_name}_noise_robustness.png'
        plt.savefig(output_filename)
        print(f"\n📈 结果图已保存至: {output_filename}")
        plt.show()


if __name__ == '__main__':
    models_to_run = ['DCC', 'ConvAE', 'DeepSVDD', 'iForest', 'VAE', 'LSTMPredictor', 'DCGAN', 'WGAN-GP']
    datasets_to_run = ['CWRU', 'SEU', 'MFPT']
    
    # ✅ 修改: 新的结果存储结构
    # results[dataset_name][model_name][noise_level] = auc_score
    results = {ds: {model: {} for model in models_to_run} for ds in datasets_to_run}

    # ✅ 修改: 主循环结构，增加噪声水平循环
    for dataset_name in datasets_to_run:
        print("\n" + "#"*80 + f"\n🚀 开始在数据集 '{dataset_name}' 上进行实验\n" + "#"*80)
        config = DATASET_CONFIGS[dataset_name]
        if not os.path.exists(config['path']):
            print(f"‼️ 警告: 路径 '{config['path']}' 不存在。跳过此数据集。")
            for model_name in models_to_run:
                for nl in NOISE_LEVELS_TO_TEST:
                    results[dataset_name][model_name][nl] = 'PathError'
            continue
        
        for model_name in models_to_run:
            for noise_level in NOISE_LEVELS_TO_TEST:
                start_time = time.time()
                print(f"\n--- [模型: {model_name} | 数据集: {dataset_name} | 噪声: {noise_level}] ---")
                
                try:
                    # 为每个实验重新加载数据，以确保测试集有正确的噪声水平
                    data_loaders = get_dataloaders(dataset_name, config, PARAMS, noise_level)
                    if data_loaders[0] is None:
                        print(f"数据加载失败，跳过此配置")
                        results[dataset_name][model_name][noise_level] = 'DataError'
                        continue
                    
                    train_loader, test_loader, train_np, test_np, y_test_np = data_loaders

                    # 如果测试集为空，无法计算AUC
                    if len(y_test_np) == 0:
                        print(f"警告: 测试集为空，无法评估。")
                        results[dataset_name][model_name][noise_level] = 'EmptyTestSet'
                        continue

                    auc = -1.0
                    if model_name == 'DCC': auc = train_evaluate_dcc(train_loader, test_loader, y_test_np, PARAMS)
                    elif model_name == 'ConvAE': auc = train_evaluate_convae(train_loader, test_loader, y_test_np, PARAMS)
                    elif model_name == 'DeepSVDD': auc = train_evaluate_deepsvdd(train_loader, test_loader, y_test_np, PARAMS)
                    elif model_name == 'iForest': auc = train_evaluate_iforest(train_np, test_np, y_test_np)
                    elif model_name == 'VAE': auc = train_evaluate_vae(train_loader, test_loader, y_test_np, PARAMS)
                    elif model_name == 'LSTMPredictor': auc = train_evaluate_lstmpredictor(train_loader, test_loader, y_test_np, PARAMS)
                    elif model_name == 'DCGAN': auc = train_evaluate_dcgan(train_loader, test_loader, y_test_np, PARAMS)
                    elif model_name == 'WGAN-GP': auc = train_evaluate_wgan_gp(train_loader, test_loader, y_test_np, PARAMS)
                    
                    results[dataset_name][model_name][noise_level] = auc
                    duration = time.time() - start_time
                    print(f"✅ [{model_name} on {dataset_name} @ Noise {noise_level}] 完成. AUC: {auc:.4f}, 耗时: {duration:.2f}s")

                except Exception as e:
                    print(f"❌ 在运行 {model_name} on {dataset_name} (Noise {noise_level}) 时发生错误: {e}")
                    traceback.print_exc()
                    results[dataset_name][model_name][noise_level] = 'Error'

    # ✅ 使用新的函数展示所有结果
    display_noise_results(results, datasets_to_run, NOISE_LEVELS_TO_TEST)
