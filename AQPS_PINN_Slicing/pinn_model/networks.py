import torch
import torch.nn as nn


class AQPS_PINN(nn.Module):
    """
    动态紧迫性感知的物理信息神经网络 (Adaptive QoS-Aware Priority Scheduling PINN)
    对应论文第 3.2 节：联合预测与可行解映射层
    """

    def __init__(self, max_slices=64):
        super(AQPS_PINN, self).__init__()
        self.max_slices = max_slices

        # 输入维度: 4个特征 (lambda, eta, buffer, psi) * 最大切片数 (64) = 256维
        input_dim = 4 * max_slices

        # 共享特征提取网络 (多层感知机 MLP)
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),  # 批归一化，加速训练并防止梯度消失
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.ReLU()
        )   # 256 to 512 to 512 to 256 to 64

        # 输出分支：输出原始的资源分配特征值 (Logits)
        self.logits_layer = nn.Linear(256, max_slices)

    def forward(self, x, mask=None):
        """
        前向传播
        :param x: 输入张量，形状为 (batch_size, 256)
        :param mask: 掩码张量 (1表示该切片活跃，0表示该切片为空/填充位)
        """
        # 1. 提取高维通用特征
        features = self.shared_net(x)

        # 2. 输出各个切片的资源分配 Logits
        logits = self.logits_layer(features)

        # 3. 动态掩码处理 (Masking)
        # 屏蔽那些用 0 填充的假切片
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9) # 负无穷

        # 4. 可行解映射层 (Feasible Solution Mapping Layer)
        # 数学上保证 ΣQ_pred = 1.0，100% 满足物理守恒约束
        Q_pred = torch.softmax(logits, dim=-1)

        return Q_pred