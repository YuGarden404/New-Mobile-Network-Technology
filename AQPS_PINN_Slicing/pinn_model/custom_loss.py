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

        if mask is not None:
            valid_elements = mask.sum() + 1e-9
            mse_loss = torch.sum(((Q_pred - Q_opt) * mask) ** 2) / valid_elements
        else:
            mse_loss = nn.MSELoss()(Q_pred, Q_opt)

        # Q -> mu -> rho -> W
        mu_pred = Q_pred * self.mu_max
        rho_pred = lambdas / (mu_pred + 1e-9)

        rho_violation = torch.relu(rho_pred - 0.95)

        rho_safe = torch.clamp(rho_pred, 0.0, 0.95)

        W_pred = (rho_safe ** 2) / (lambdas * (1.0 - rho_safe) + 1e-9) * 1000.0

        urgency = psis * (1.0 + buffers / (self.buffer_max + 1e-9)) * (1.0 + 1.0 / (etas + self.epsilon))

        qos_violation = torch.relu(W_pred - etas) / (etas + 1e-9)

        if mask is not None:
            qos_penalty_term = torch.sum((urgency * qos_violation) * mask) / valid_elements
            rho_penalty_term = torch.sum((urgency * rho_violation) * mask) / valid_elements
        else:
            qos_penalty_term = torch.mean(urgency * qos_violation)
            rho_penalty_term = torch.mean(urgency * rho_violation)

        qos_penalty = qos_penalty_term + rho_penalty_term
        total_loss = mse_loss + self.alpha_qos * qos_penalty

        return total_loss, mse_loss, qos_penalty