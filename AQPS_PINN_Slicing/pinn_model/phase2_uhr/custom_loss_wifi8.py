import torch
import torch.nn as nn
import json
import os


class AQPS_Loss_WiFi8(nn.Module):
    """
    面向 Wi-Fi 8 的跨层物理信息损失函数 (Cross-Layer PINN Loss for Wi-Fi 8)
    引入了 DRU/CRU 的物理信道容量模型，实现物理法则引导的弱监督学习
    """

    def __init__(self, config_path,eta_max,psi_max, dru_efficiency_loss=0.10):
        super(AQPS_Loss_WiFi8, self).__init__()
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

        # DRU 物理开销系数 (Pilot Overhead & MCS Downgrade)
        # 在 OFDMA 中，基站为了解析接收到的信号，必须在数据中插入“导频子载波（Pilot Subcarriers）”
        # 根据 IEEE 802.11ax/be 中关于分布式交织（Interleaved / Distributed OFDMA）的早期研究
        # 将连续子载波打散后，为了维持相同的误码率（BER）
        # 物理层头部（PHY Header）和导频的开销通常会增加 10% 到 15%
        self.dru_efficiency_loss = dru_efficiency_loss  # 可能为0.10 0.15 0.20

        # 频率分集增益因子 (Frequency Diversity Factor)
        # 越小代表 DRU 抗干扰能力越强
        # 现代 Wi-Fi 使用 LDPC（低密度奇偶校验码）
        # LDPC 极其害怕“突发连续错误（Burst Errors，即 CRU 遭遇的情况）”
        # 但对于“均匀分布的随机擦除（Random Erasures，即 DRU 遭遇的情况）”
        # 只要错误率在 10% 左右，LDPC 可以完美纠错
        # 窄带干扰对 DRU 的实际破坏力，会被物理层的频率分集和 LDPC 编码大幅稀释
        # 所以 0.1 只是一个保守的缩放因子
        self.dru_diversity_factor = 0.10

        # MAC 层控制信令开销 / 模式切换惩罚权重
        self.lambda_dru_penalty = 0.05

    def forward(self, Q_pred, dru_prob, Q_opt, features_normalized, mask=None):
        """
        前向计算物理损失
        """
        lambdas = features_normalized[:, 0, :] * self.mu_max
        etas = features_normalized[:, 1, :] * self.eta_max
        buffers = features_normalized[:, 2, :] * self.buffer_max
        psis = features_normalized[:, 3, :] * self.psi_max
        interferences = features_normalized[:, 4, :]

        if mask is not None:
            valid_elements = mask.sum() + 1e-9
            mse_loss = torch.sum(((Q_pred - Q_opt) * mask) ** 2) / valid_elements
        else:
            mse_loss = nn.MSELoss()(Q_pred, Q_opt)

        # Wi-Fi 8 物理层信道容量重建 (CRU vs DRU)

        # 基础物理容量
        C_total = Q_pred * self.mu_max

        # 使用 CRU (连续资源)
        # 特点：频谱效率 100%，但在遭遇窄带干扰时极其脆弱 (容量直接按干扰比例扣除)
        C_cru = C_total * (1.0 - interferences)

        # 使用 DRU (分布式资源)
        # 特点：有 10% 的物理打散开销 (乘以 0.9)，但几乎免疫窄带干扰 (干扰影响降至原来的十分之一)
        C_dru = C_total * (1 - self.dru_efficiency_loss) * (1.0 - interferences * self.dru_diversity_factor)

        # AI 根据它对环境的判断，输出的实际期望容量
        C_actual = (1.0 - dru_prob) * C_cru + dru_prob * C_dru

        # Q -> mu -> rho -> W
        rho_pred = lambdas / (C_actual + 1e-9)

        rho_violation = torch.relu(rho_pred - 0.95)

        rho_safe = torch.clamp(rho_pred, 0.0, 0.95)

        W_pred = (rho_safe ** 2) / (lambdas * (1.0 - rho_safe) + 1e-9) * 1000.0

        urgency = psis * (1.0 + buffers / (self.buffer_max + 1e-9)) * (1.0 + 1.0 / (etas + self.epsilon))

        qos_violation = torch.relu(W_pred - etas) / (etas + 1e-9)

        # 为了防止 AI 偷懒，无论有没有干扰都无脑开 DRU，给开启 DRU 加上一个极小的成本
        dru_overhead_penalty = self.lambda_dru_penalty * dru_prob

        if mask is not None:
            qos_penalty_term = torch.sum((urgency * qos_violation) * mask) / valid_elements
            rho_penalty_term = torch.sum((urgency * rho_violation) * mask) / valid_elements
            dru_penalty_term = torch.sum(dru_overhead_penalty * mask) / valid_elements
        else:
            qos_penalty_term = torch.mean(urgency * qos_violation)
            rho_penalty_term = torch.mean(urgency * rho_violation)
            dru_penalty_term = torch.mean(dru_overhead_penalty)

        qos_penalty = qos_penalty_term + rho_penalty_term
        total_loss = mse_loss + self.alpha_qos * qos_penalty + dru_penalty_term

        return total_loss, mse_loss, qos_penalty