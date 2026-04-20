import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from pinn_model.networks import AQPS_PINN
from pinn_model.custom_loss import AQPS_Loss


# ================= 1. 数据集加载器 (PyTorch Dataset) =================
class SlicingDataset(Dataset):
    """
    负责把我们在 dataset_builder.py 里生成的 .pt 文件喂给 GPU
    """

    def __init__(self, data_path,config_path, eta_max_prior=None, psi_max_prior=None):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据集文件: {data_path}，请先运行 dataset_builder.py")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 获取物理系统的绝对极限容量
        self.mu_max = config["system_parameters"]["mu_max"]
        # self.buffer_max = config["system_parameters"]["max_buffer_size_bytes"]
        b_max_dict = config["slice_definitions"]["B_max_packets"]
        self.buffer_max = float(max(b_max_dict.values()))


        print(f"正在加载数据集: {data_path}...")
        data = torch.load(data_path)

        X_raw = data['X'].view(-1, 4, 64)

        if eta_max_prior is not None and psi_max_prior is not None:
            self.eta_max = eta_max_prior
            self.psi_max = psi_max_prior
            print(f"[归一化参数] (来自训练集) Eta_max: {self.eta_max:.2f} ms, Psi_max: {self.psi_max:.2f}")
        else:
            self.eta_max = X_raw[:, 1, :].max().item() + 1e-5
            self.psi_max = X_raw[:, 3, :].max().item() + 1e-5
            print(f"[归一化参数] μ_max: {self.mu_max}, Buffer_max: {self.buffer_max}")
            print(f"[归一化参数] 提取到全局 Eta_max: {self.eta_max:.2f} ms, Psi_max: {self.psi_max:.2f}")

        # Min-Max 归一化
        X_normalized = torch.zeros_like(X_raw)

        X_normalized[:, 0, :] = X_raw[:, 0, :] / self.mu_max
        X_normalized[:, 1, :] = X_raw[:, 1, :] / self.eta_max
        X_normalized[:, 2, :] = X_raw[:, 2, :] / self.buffer_max
        X_normalized[:, 3, :] = X_raw[:, 3, :] / self.psi_max

        self.X = X_normalized   # 形状: [N, 4, 64]
        self.Y = data['Y']      # 形状: [N, 64]


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ================= 2. 主训练循环 =================
def train_model():
    print(f"\n启动 AQPS-PINN 物理信息神经网络训练模型")

    # --- 基础配置 ---
    config_path = os.path.join(current_dir, "problem_descriptors/slicing_params.json")
    train_data_path = os.path.join(current_dir, "dataset/train_data.pt")
    test_data_path = os.path.join(current_dir, "dataset/test_data.pt")
    checkpoint_dir = os.path.join(current_dir, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # --- 超参数设置 ---
    BATCH_SIZE = 64
    EPOCHS = 450
    LEARNING_RATE = 1e-3
    MAX_SLICES = 64

    # 自动选择设备 (如果有 NVIDIA 显卡就用 GPU，否则用 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # --- 加载数据 ---
    train_dataset = SlicingDataset(train_data_path,config_path)
    test_dataset = SlicingDataset(test_data_path,config_path,
                                  eta_max_prior=train_dataset.eta_max,
                                  psi_max_prior=train_dataset.psi_max)  # 最大值继承自训练集，防止数据泄露

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 初始化模型与损失函数 ---
    model = AQPS_PINN(max_slices=MAX_SLICES).to(device)
    criterion = AQPS_Loss(
        config_path=config_path,
        eta_max=train_dataset.eta_max,
        psi_max=train_dataset.psi_max
    ).to(device)

    # 使用 Adam 优化器 (深度学习最经典的自适应优化器)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # 学习率调度器：如果连续 5 个 Epoch 验证集 Loss 不下降，就把学习率砍半，帮模型做精细微调
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')

    count_promote = 0

    # ================= 开始纪元 (Epoch) 循环 =================
    start_time = time.time()

    for epoch in range(EPOCHS):
        # ---------------- [训练阶段] ----------------
        model.train()  # 开启训练模式 (启用 BatchNorm 和 Dropout)
        train_total_loss, train_mse, train_qos = 0.0, 0.0, 0.0

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)


            mask_batch = (batch_X[:, 0, :] > 0).float()

            # 梯度清零
            optimizer.zero_grad()

            # 前向推理
            q_pred = model(batch_X, mask=mask_batch)

            # 计算物理损失
            loss, mse, qos = criterion(q_pred, batch_Y, batch_X, mask=mask_batch)

            # 反向传播与权重更新
            loss.backward()
            optimizer.step()

            # 累计数据
            train_total_loss += loss.item()
            train_mse += mse.item()
            train_qos += qos.item()

        # 计算平均 Loss
        num_batches = len(train_loader)
        avg_train_loss = train_total_loss / num_batches
        avg_train_mse = train_mse / num_batches
        avg_train_qos = train_qos / num_batches

        # ---------------- [验证阶段] ----------------
        model.eval()
        val_total_loss, val_mse, val_qos = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                mask_batch = (batch_X[:, 0, :] > 0).float()

                q_pred = model(batch_X, mask=mask_batch)
                loss, mse, qos = criterion(q_pred, batch_Y, batch_X, mask=mask_batch)

                val_total_loss += loss.item()
                val_mse += mse.item()
                val_qos += qos.item()

        avg_val_loss = val_total_loss / len(test_loader)
        avg_val_mse = val_mse / len(test_loader)
        avg_val_qos = val_qos / len(test_loader)

        # 触发学习率衰减
        scheduler.step(avg_val_loss)

        # ---------------- [日志与保存] ----------------
        # 每 10 个 Epoch 打印一次详细日志
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\n[Epoch {epoch + 1:03d}/{EPOCHS}]")
            print(f"Train | Total Loss: {avg_train_loss:.4f} (MSE: {avg_train_mse:.4f}, QoS: {avg_train_qos:.4f})")
            print(f"Valid | Total Loss: {avg_val_loss:.4f}   (MSE: {avg_val_mse:.4f}, QoS: {avg_val_qos:.4f})")

        # 保存验证集上表现最好的模型
        if avg_val_loss < best_val_loss:
            count_promote+=1
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_aqps_pinn.pth"))
            print(f"[保存] 模型已更新至最低 Valid Loss: {best_val_loss:.4f}")

    # ================= 训练结束 =================
    total_time = (time.time() - start_time) / 60
    print(f"\n训练完成，总耗时: {total_time:.2f} 分钟")
    print(f"优化次数: {count_promote} 次")
    best_model_path = os.path.join(checkpoint_dir, "best_aqps_pinn.pth")
    model.load_state_dict(torch.load(best_model_path))

    print(f"最佳模型已封存并重新加载至内存: {best_model_path}")


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    train_model()