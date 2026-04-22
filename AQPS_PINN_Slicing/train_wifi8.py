import torch
from torch.utils.data import Dataset, DataLoader
import os
import time
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))

from pinn_model.phase2_uhr.networks_wifi8 import AQPS_PINN_WiFi8
from pinn_model.phase2_uhr.custom_loss_wifi8 import AQPS_Loss_WiFi8


class SlicingDataset(Dataset):
    """
    Wi-Fi 8 数据加载器：负责解析包含物理层干扰的 5 维数据张量
    """

    def __init__(self, data_path, config_path, eta_max_prior=None, psi_max_prior=None):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件路径错误: {data_path}，请先运行 dataset_builder.py")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 获取物理系统的绝对极限容量
        self.mu_max = config["system_parameters"]["mu_max"]
        # self.buffer_max = config["system_parameters"]["max_buffer_size_bytes"]
        b_max_dict = config["slice_definitions"]["B_max_packets"]
        self.buffer_max = float(max(b_max_dict.values()))


        print(f"正在加载数据集: {data_path}...")
        data = torch.load(data_path)

        X_raw = data['X'].view(-1, 5, 64)

        if eta_max_prior is not None and psi_max_prior is not None:
            self.eta_max = eta_max_prior
            self.psi_max = psi_max_prior
            print(f"[归一化参数] Eta_max: {self.eta_max:.2f} ms, Psi_max: {self.psi_max:.2f}")
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
        X_normalized[:, 4, :] = X_raw[:, 4, :]

        self.X = X_normalized   # 形状: [N, 5, 64]
        self.Y = data['Y']      # 形状: [N, 64]


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_model():
    print(f"\n启动 AQPS-PINN (Wi-Fi 8 跨层协同版) 神经网络训练模型")

    config_path = os.path.join(current_dir, "problem_descriptors/slicing_params.json")
    train_data_path = os.path.join(current_dir, "dataset/phase2_uhr/train_uhr.pt")
    test_data_path = os.path.join(current_dir, "dataset/phase2_uhr/test_uhr.pt")
    checkpoint_dir = os.path.join(current_dir, "checkpoints/phase2_uhr")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 参数设置
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 5e-4
    MAX_SLICES = 64
    FINAL_ALPHA = 0.005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    train_dataset = SlicingDataset(train_data_path,config_path)
    test_dataset = SlicingDataset(test_data_path,config_path,
                                  eta_max_prior=train_dataset.eta_max,
                                  psi_max_prior=train_dataset.psi_max)  # 最大值继承自训练集，防止数据泄露

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AQPS_PINN_WiFi8(max_slices=MAX_SLICES, input_features=5).to(device)
    criterion = AQPS_Loss_WiFi8(
        config_path=config_path,
        eta_max=train_dataset.eta_max,
        psi_max=train_dataset.psi_max
    ).to(device)

    # 使用 Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 学习率调度器：如果连续 5 个 Epoch 验证集 Loss 不下降，就把学习率砍半，帮模型做精细微调
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')

    count_promote = 0

    start_time = time.time()

    for epoch in range(EPOCHS):
        # 让 alpha_qos 随着 epoch 从 0.001 慢慢线性增长到 0.005
        # 前期宽容，让模型学 MSE；后期严格，保证QoS
        current_alpha = 0.001 + (FINAL_ALPHA - 0.0001) * (epoch / EPOCHS)
        criterion.alpha_qos = current_alpha

        model.train()
        train_total_loss, train_mse, train_qos = 0.0, 0.0, 0.0

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            mask_batch = (batch_X[:, 0, :] > 0).float()

            optimizer.zero_grad()

            q_pred, dru_prob = model(batch_X, mask=mask_batch)

            loss, mse, qos = criterion(q_pred, dru_prob, batch_Y, batch_X, mask=mask_batch)

            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            train_mse += mse.item()
            train_qos += qos.item()

        num_batches = len(train_loader)
        avg_train_loss = train_total_loss / num_batches
        avg_train_mse = train_mse / num_batches
        avg_train_qos = train_qos / num_batches

        model.eval()
        criterion.alpha_qos = FINAL_ALPHA
        val_total_loss, val_mse, val_qos = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                mask_batch = (batch_X[:, 0, :] > 0).float()

                q_pred, dru_prob = model(batch_X, mask=mask_batch)
                loss, mse, qos = criterion(q_pred, dru_prob, batch_Y, batch_X, mask=mask_batch)

                val_total_loss += loss.item()
                val_mse += mse.item()
                val_qos += qos.item()

        avg_val_loss = val_total_loss / len(test_loader)
        avg_val_mse = val_mse / len(test_loader)
        avg_val_qos = val_qos / len(test_loader)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\n[Epoch {epoch + 1:03d}/{EPOCHS}]")
            print(f"Train | Total Loss: {avg_train_loss:.4f} (MSE: {avg_train_mse:.4f}, QoS: {avg_train_qos:.4f})")
            print(f"Valid | Total Loss: {avg_val_loss:.4f} (MSE: {avg_val_mse:.4f}, QoS: {avg_val_qos:.4f})")

        if avg_val_loss < best_val_loss:
            count_promote+=1
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_aqps_pinn_wifi8.pth"))
            print(f"[保存] 模型已更新至最低 Valid Loss: {best_val_loss:.4f}")

    # 训练结束
    total_time = (time.time() - start_time) / 60
    print(f"\n训练完成，总耗时: {total_time:.2f} 分钟")
    print(f"优化次数: {count_promote} 次")
    best_model_path = os.path.join(checkpoint_dir, "best_aqps_pinn_wifi8.pth")
    model.load_state_dict(torch.load(best_model_path))

    print(f"Wi-Fi 8 最佳模型已封存并重新加载至内存: {best_model_path}")


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    train_model()