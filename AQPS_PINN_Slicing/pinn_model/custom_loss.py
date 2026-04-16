import torch
import torch.nn as nn
import json
import os


class AQPS_Loss(nn.Module):
    """
    动态紧迫性驱动的 QoS 感知损失函数 (Dynamic Urgency-Driven QoS-Aware Loss)
    对应论文 3.3 节与 3.4 节的核心创新
    """

    def __init__(self, config_path="../problem_descriptors/slicing_params.json"):
        super(AQPS_Loss, self).__init__()
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件路径错误: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.mu_max = self.config["system_parameters"]["mu_max"]
        self.epsilon = self.config["system_parameters"]["epsilon"]
        self.alpha_qos = self.config["training_hyperparameters"]["alpha_qos"]

        slice_cfg = self.config["slice_definitions"]
        self.psi_u, self.psi_e, self.psi_b = slice_cfg["psi"]["URLLC"], slice_cfg["psi"]["eMBB"], slice_cfg["psi"]["BE"]
        self.wqos_u, self.wqos_e, self.wqos_b = slice_cfg["W_qos_ms"]["URLLC"], slice_cfg["W_qos_ms"]["eMBB"], slice_cfg["W_qos_ms"]["BE"]
        self.bmax_u, self.bmax_e, self.bmax_b = slice_cfg["B_max_packets"]["URLLC"], slice_cfg["B_max_packets"]["eMBB"], slice_cfg["B_max_packets"]["BE"]

    def forward(self, Q_pred, Q_opt, features, mask=None):
        """
        前向计算物理损失
        :param Q_pred: AI 预测的分配比例 (batch_size, max_slices)
        :param Q_opt: 传统优化器给出的标准答案 (batch_size, max_slices)
        :param features: 环境特征 (batch_size, 4, max_slices) -> 行分别对应 λ, η, B, ψ
        :param mask: 有效切片掩码
        """
        lambdas = features[:, 0, :]
        etas = features[:, 1, :]
        buffers = features[:, 2, :]
        psis = features[:, 3, :]

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

        # --- 3. 动态提取底层物理约束 (消除硬编码，根据加载的 JSON 动态映射) ---
        W_qos_tensor = torch.ones_like(psis) * self.wqos_b  # 默认填 BE 的延迟
        W_qos_tensor[psis == self.psi_u] = self.wqos_u  # 匹配 URLLC
        W_qos_tensor[psis == self.psi_e] = self.wqos_e  # 匹配 eMBB

        B_max_tensor = torch.ones_like(psis) * self.bmax_b  # 默认填 BE 的缓冲
        B_max_tensor[psis == self.psi_u] = self.bmax_u  # 匹配 URLLC
        B_max_tensor[psis == self.psi_e] = self.bmax_e  # 匹配 eMBB

        # --- 4. 动态紧迫性计算 (Dynamic Urgency) ---
        # 对应论文 3.4 节核心公式: X(t) = psi * (1 + B/B_max) * (1 + 1/(eta + epsilon))
        # 当信道极差 (eta很小) 或 缓冲区快满 (B接近B_max) 时，权重呈指数级放大！
        urgency = psis * (1.0 + buffers / (B_max_tensor + 1e-9)) * (1.0 + 1.0 / (etas + self.epsilon))

        # --- 5. QoS 感知惩罚计算 ---
        # 只有当 AI 预测的时延 W_pred 超过了容忍度 W_qos_tensor 时，才触发惩罚
        qos_violation = torch.relu(W_pred - W_qos_tensor)

        if mask is not None:
            qos_penalty = torch.sum((urgency * qos_violation) * mask) / valid_elements
        else:
            qos_penalty = torch.mean(urgency * qos_violation)

        # 最终的联合损失函数
        total_loss = mse_loss + self.alpha_qos * qos_penalty

        return total_loss, mse_loss, qos_penalty


# ================= 测试代码 =================
if __name__ == "__main__":
    print("🚀 正在测试物理信息可微损失函数 (AQPS-Loss)...")

    # 模拟一张极其危险的 URLLC 考卷 (1张考卷, 4个特征, 1个切片)
    # 必须严格按照 (batch_size, 4, max_slices) 的形状构建张量
    features = torch.tensor([[
        [400.0],  # lambda (特征0)
        [0.1],  # eta    (特征1)
        [48.0],  # buffer (特征2)
        [10.0]  # psi    (特征3)
    ]])

    # AI 犯傻了，只给 URLLC 分了 25% 的资源 (mu = 1720*0.25 = 430)
    # rho = 400/430 = 0.93，濒临崩溃，时延绝对会远超 1ms！
    Q_pred = torch.tensor([[0.25]])
    Q_opt = torch.tensor([[0.80]])  # 优化器给的正确答案应该是 80%

    loss_fn = AQPS_Loss()
    total_loss, mse, qos = loss_fn(Q_pred, Q_opt, features)

    print(f"✅ Loss 计算图构建成功！")
    print(f"基础拟合误差 (MSE Loss): {mse.item():.4f}")
    print(f"物理违约惩罚 (QoS Penalty): {qos.item():.4f}")
    print(f"总损失 (Total Loss): {total_loss.item():.4f}")

    if qos.item() > mse.item():
        print("\n💡 [论文机制验证成功]: 在极端拥塞与极差信道下，底层物理违约引发了巨大的惩罚梯度！")
        print("这股巨大的梯度将顺着计算图反向传播，狠狠地‘扇’AI一巴掌，逼它下次给 URLLC 分配更多资源！")