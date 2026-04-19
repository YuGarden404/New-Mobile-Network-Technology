import torch
import torch.nn as nn
import json
import os


class AQPS_Loss(nn.Module):
    """
    动态紧迫性驱动的 QoS 感知损失函数 (Dynamic Urgency-Driven QoS-Aware Loss)
    对应论文 3.3 节与 3.4 节的核心创新
    """

    def __init__(self, config_path,eta_max,psi_max):
        super(AQPS_Loss, self).__init__()
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件路径错误: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.mu_max = self.config["system_parameters"]["mu_max"]
        # self.buffer_max = self.config["system_parameters"]["max_buffer_size_bytes"]
        b_max_dict = self.config["slice_definitions"]["B_max_packets"]
        self.buffer_max = float(max(b_max_dict.values()))

        self.epsilon = self.config["system_parameters"]["epsilon"]
        self.alpha_qos = self.config["training_hyperparameters"]["alpha_qos"]

        self.eta_max = eta_max
        self.psi_max = psi_max

    def forward(self, Q_pred, Q_opt, features_normalized, mask=None):
        """
        前向计算物理损失
        :param Q_pred: AI 预测的分配比例 (batch_size, max_slices)
        :param Q_opt: 传统优化器给出的标准答案 (batch_size, max_slices)
        :param features: 环境特征 (batch_size, 4, max_slices) -> 行分别对应 λ, η, B, ψ
        :param mask: 有效切片掩码
        """
        lambdas = features_normalized[:, 0, :] * self.mu_max
        etas = features_normalized[:, 1, :] * self.eta_max
        buffers = features_normalized[:, 2, :] * self.buffer_max
        psis = features_normalized[:, 3, :] * self.psi_max

        # --- 1. 基础回归损失 (MSE) ---
        # AI 需要尽量模仿传统优化器的答案
        if mask is not None:
            valid_elements = mask.sum() + 1e-9
            mse_loss = torch.sum(((Q_pred - Q_opt) * mask) ** 2) / valid_elements
        else:
            mse_loss = nn.MSELoss()(Q_pred, Q_opt)

        # --- 2. 物理信息可微排队层 (Differentiable Queueing Layer) ---
        # 对应论文 3.3 节: Q -> mu -> rho -> W 的全链条可微推导
        mu_pred = Q_pred * self.mu_max
        rho_pred = lambdas / (mu_pred + 1e-9)

        # 为了反向传播的稳定性，截断 rho，防止底层数学出现负数或除零爆炸
        rho_safe = torch.clamp(rho_pred, 0.0, 0.95)

        # 计算预测分配比例下，系统会产生的真实排队时延 W (ms)
        W_pred = (rho_safe ** 2) / (lambdas * (1.0 - rho_safe) + 1e-9) * 1000.0

        # # --- 3. 动态提取底层物理约束 (消除硬编码，根据加载的 JSON 动态映射) ---
        # W_qos_tensor = torch.ones_like(psis) * self.wqos_b  # 默认填 BE 的延迟
        # W_qos_tensor[psis == self.psi_u] = self.wqos_u  # 匹配 URLLC
        # W_qos_tensor[psis == self.psi_e] = self.wqos_e  # 匹配 eMBB
        #
        # B_max_tensor = torch.ones_like(psis) * self.bmax_b  # 默认填 BE 的缓冲
        # B_max_tensor[psis == self.psi_u] = self.bmax_u  # 匹配 URLLC
        # B_max_tensor[psis == self.psi_e] = self.bmax_e  # 匹配 eMBB

        # --- 4. 动态紧迫性计算 (Dynamic Urgency) ---
        # 对应论文 3.4 节核心公式: X(t) = psi * (1 + B/B_max) * (1 + 1/(eta + epsilon))
        # 当信道极差 (eta很小) 或 缓冲区快满 (B接近B_max) 时，权重呈指数级放大！
        urgency = psis * (1.0 + buffers / (self.buffer_max + 1e-9)) * (1.0 + 1.0 / (etas + self.epsilon))

        # --- 5. QoS 感知惩罚计算 ---
        # 只有当 AI 预测的时延 W_pred 超过了容忍度 W_qos_tensor 时，才触发惩罚
        qos_violation = torch.relu(W_pred - etas)

        if mask is not None:
            qos_penalty = torch.sum((urgency * qos_violation) * mask) / valid_elements
        else:
            qos_penalty = torch.mean(urgency * qos_violation)

        # 最终的联合损失函数
        total_loss = mse_loss + self.alpha_qos * qos_penalty

        return total_loss, mse_loss, qos_penalty


# ================= 测试代码 =================
if __name__ == "__main__":
    import sys

    print("正在测试物理信息可微损失函数 (AQPS-Loss)...")

    # 1. 解决路径依赖，获取真实的物理配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    config_path = os.path.join(parent_dir, "problem_descriptors/slicing_params.json")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    mu_max = config["system_parameters"]["mu_max"]
    # buffer_max = config["system_parameters"]["max_buffer_size_bytes"]
    b_max_dict = config["slice_definitions"]["B_max_packets"]
    buffer_max = float(max(b_max_dict.values()))

    # 2. 模拟我们在 train.py 里从 Dataset 提取到的全局物理上限
    mock_eta_max = 100.0  # 假设整个数据集最大的延迟是 100ms
    mock_psi_max = 10.0  # 假设权重最大是 10

    # 3. 实例化 Loss (补齐缺失的参数！)
    loss_fn = AQPS_Loss(config_path=config_path, eta_max=mock_eta_max, psi_max=mock_psi_max)

    # 4. 模拟一张极其危险的 URLLC 考卷 (必须送入 0~1 的归一化特征！)
    # 真实物理含义: lambda=400, eta=1.0ms, buffer=48, psi=10
    # (注：你的eta=0.1ms过于严苛可能会让rho方程瞬间炸出天文数字，这里微调为1.0ms测试)
    features_normalized = torch.tensor([[
        [400.0 / mu_max],  # 特征0: lambda 比例
        [1.0 / mock_eta_max],  # 特征1: eta 比例
        [48.0 / buffer_max],  # 特征2: buffer 比例
        [10.0 / mock_psi_max]  # 特征3: psi 比例
    ]])

    # AI 犯傻了，只给 URLLC 分了 25% 的资源 (mu = 1720*0.25 = 430)
    # rho = 400/430 = 0.93，排队时延绝对会远超 1.0ms！
    Q_pred = torch.tensor([[0.25]])
    Q_opt = torch.tensor([[0.80]])  # 标准答案是 80%

    # 模拟 mask，表示这 1 个切片是有效的
    mask = torch.tensor([[1.0]])

    total_loss, mse, qos = loss_fn(Q_pred, Q_opt, features_normalized, mask)
    print(f"Loss 计算图构建成功！")
    print(f"基础拟合误差 (MSE Loss): {mse.item():.4f}")
    print(f"物理违约惩罚 (QoS Penalty): {qos.item():.4f}")
    print(f"总损失 (Total Loss): {total_loss.item():.4f}")

    if qos.item() > mse.item():
        print("\n[论文机制验证成功]: 在极端拥塞与极差信道下，底层物理违约引发了巨大的惩罚梯度")
    else:
        print("\n[警告]: QoS 惩罚没有起效，请检查超参数 alpha_qos")