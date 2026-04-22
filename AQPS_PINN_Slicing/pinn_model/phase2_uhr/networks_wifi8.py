import torch
import torch.nn as nn


class AQPS_PINN_WiFi8(nn.Module):
    """
    面向 Wi-Fi 8 的多任务物理信息神经网络 (Multi-Task PINN for Wi-Fi 8)
    不仅输出 RU 带宽分配比例，同时输出物理层抗干扰 DRU 模式的激活概率
    """

    def __init__(self, max_slices=64,input_features=5):
        super(AQPS_PINN_WiFi8, self).__init__()
        self.max_slices = max_slices

        # 输入维度: 5个特征 (lambda, eta, buffer, psi, interference) * 最大切片数 (64) = 320维
        input_dim = input_features * max_slices

        # 共享特征提取网络 (多层感知机 MLP)
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),    # 从 0.1 改为 0.3，让训练时每次神经元失活 30%

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),    # 从 0.1 改为 0.3

            nn.Linear(512, 256),
            nn.ReLU()
        )
        # 双头输出
        # 计算 RU 分配的 Logits
        self.q_logits_layer = nn.Linear(256, max_slices)

        # 计算 DRU 概率的 Logits
        self.dru_logits_layer = nn.Linear(256, max_slices)


    def forward(self, x, mask=None):
        """
        前向传播
        :param x: 输入张量，形状为 (batch_size, 5, 64)
        :param mask: 掩码张量 (1表示该切片活跃，0表示该切片为空/填充位)
        """
        # 形状对齐：将 [N, 5, max_slices] 展平为 [N, 5 * max_slices] = [N, 320]
        x_flat = x.view(x.size(0), -1)
        features = self.shared_net(x_flat)

        # 带宽分配比例 Q_pred
        q_logits = self.q_logits_layer(features)

        if mask is not None:
            q_logits = q_logits.masked_fill(mask == 0, -1e9) # 负无穷

        # 保证 ΣQ_pred = 1.0，满足物理守恒约束
        Q_pred = torch.softmax(q_logits, dim=-1)

        # DRU 激活概率 dru_prob
        dru_logits = self.dru_logits_layer(features)
        dru_prob = torch.sigmoid(dru_logits)
        if mask is not None:
            dru_prob = dru_prob * mask

        return Q_pred, dru_prob
"""
Q_pred 用 Softmax 因为切片的带宽加起来必须等于 100%，是竞争关系。给 URLLC 多分一点，eMBB 就少一点。
dru_prob 用 Sigmoid 因为 DRU 的开启是每个设备独立的决策。切片 1 开启 DRU，完全不影响切片 2 是否开启。Sigmoid 会给每一个切片打出一个 0 到 1 之间的概率值。
"""

"""
使用MLP不使用其他架构的原因：

1. 为什么不用 CNN (卷积神经网络)
  - CNN 擅长处理有“局部空间相邻关系”的数据（如图片像素）
    但 64 个切片在物理空间上并没有固定的相邻关系
    用 CNN 提取局部特征毫无物理意义
2. 为什么不用 RNN/LSTM (循环神经网络)
    RNN 擅长处理时间序列（如语音、文本）
    但现在的输入是并发 64 个切片的状态
    没有先后顺序，因此不需要 RNN
3. 为什么不用 Transformer
    Transformer 的 Self-Attention 可以很好地捕捉 64 个切片之间互相抢夺资源的竞争关系
    但是 Attention 的计算复杂度是 O(N²)
    在 Wi-Fi 调度中，要求 1 ms
    Transformer 复杂的矩阵 Q, K, V 运算极难在微秒级完成
4. 为什么用 MLP
  - 低延迟
  - MLP 的全连接特性对于做“全局资源零和博弈（和为 100%）”非常重要
  - 根据通用近似定理 (Universal Approximation Theorem)
    只要神经元足够多，MLP 能拟合任何复杂的非线性函数
    
"""