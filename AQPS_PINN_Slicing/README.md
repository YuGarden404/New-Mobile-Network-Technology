# 项目文档

## 第一阶段

### 阶段总结与核心贡献
在本项目的第一阶段，我们聚焦于下一代无线局域网（WLAN）中面临的计算复杂性灾难：**如何在极短的时间内（毫秒级），在资源受限的物理层完成满足绝对确定性延迟（Deterministic Latency）的网络切片资源调度？**

传统的非线性数学求解器（如基于 SLSQP 的 QCQP 优化）在面对大规模并发切片时，其 $\mathcal{O}(N^3)$ 的时间复杂度会导致耗时呈指数级爆炸（64切片下耗时高达 8 秒），彻底丧失了实时调度的工程可行性。为此，我们提出了 **AQPS-PINN (应用定义的 QoS 感知物理信息神经网络)** 架构，实现了对传统算法的极大提升。

**核心技术亮点：**
1. **PHY-MAC 跨层物理规则嵌入：** 我们没有使用传统的黑盒式强化学习，而是将通信物理层的香农信道极限（$\mu_{max}$）与非线性 M/M/1 排队论延迟边界（$W \propto \frac{\rho}{1-\rho}$）编码到了神经网络的自定义 Loss 函数中（`custom_loss.py`）。
2. **绝对延迟红线守卫：** 强迫 AI 在训练时学习底层物理法则。一旦模型给出的分配策略突破了 URLLC 切片的极低延迟约束限度（$\eta$），物理引擎将施加剧烈的指数级惩罚。
3. **零样本扩展性 (Zero-Shot Scalability)：** 模型在训练时最高仅见过 32 个并发切片的场景，但在测试集面对从未见过的 64 切片超大规模并发时，依然展现出了惊人的分布外泛化能力。
4. **8000倍的 $\mathcal{O}(1)$ 极速推理：** 实验（`evaluate.py`）证明，AQPS-PINN 将 64 切片的调度时间从 8299ms 压缩至 1.1ms，在牺牲极微小物理精度的前提下，实现了物理层 OFDMA 资源分配的实时可用。

### 第一阶段项目结构图

```
AQPS_PINN_Slicing/
│
├── checkpoints/                    # 存放最佳模型
│   └── best_aqps_pinn.pth          # 存放最佳模型参数
│
├── problem_descriptors/            # 存放所有超参数和网络配置
│   └── slicing_params.json         # 全局物理参数、SLA阈值、优先级的 JSON 配置文件
│
├── data_generation/                # 数据引擎：因为我们要训练 AI，必须有数据
│   ├── traffic_simulator.py        # 模拟生成真实的泊松流量、突发信道衰落和缓冲区积压
│   └── dataset_builder.py          # 离线调用优化器，生成供 AI 学习的 (X, Y_opt) 数据集
│
├── dataset/                        # 数据集
│   ├── test_data.pt                # 测试数据集
│   └── train_data.pt               # 训练数据集
│
├── non_linear_optimization/        # 传统数学求解器引擎
│   ├── scipy_solver.py             # 使用 SciPy SLSQP 实现的 QCQP 求解器 # 已弃用
│   └── gekko_solver.py             # 使用 Gekko SLSQP 实现的 QCQP 求解器
│
├── pinn_model/                     # 物理信息神经网络引擎
│   ├── networks.py                 # 多任务 DNN 架构
│   └── custom_loss.py              # 可微 M/M/1 排队层与动态紧迫性 Loss
│
├── results/                        # 实验结果可视化脚本
│   ├── evaluation_results.csv      # 评估结果的表格
│   ├── scalability_test.pdf        # 绘制 纯数学 vs 纯AI 的耗时与 Loss 对比
│   └── scalability_test.png        # 绘制 纯数学 vs 纯AI 的耗时与 Loss 对比
│
├── test/                           # 测试代码
│   └── evaluate.py                 # 执行在线实时调度与对比测试的入口脚本
│
├── train.py                        # 执行离线训练的入口脚本
├── requirements.txt                # 运行环境依赖 (torch, scipy, matplotlib等)
└── README.md                       # 项目文档
```