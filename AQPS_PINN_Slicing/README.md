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

## 第二阶段：面向 Wi-Fi 8 (802.11bn) 的抗干扰跨层调度与 DRU 自主觉醒

### 阶段总结与核心贡献
在第二阶段，我们将物理环境的复杂度提升到了真实的工业界水平。Wi-Fi 8 标准引入了**分布式资源单元 (DRU)** 来应对现实空间中突发的“窄带干扰”。然而，传统的连续变量数学求解器（如 Gekko/SLSQP）在面对混合整数非线性规划（MINLP，即连续带宽划分 + 离散 DRU 开关）时，计算彻底崩溃。

面对包含强干扰的现实信道，传统 CRU 分配会导致系统容量断崖式下跌，引发极其惨烈的排队延迟爆炸（QoS Penalty 破千）。为此，我们对 AQPS-PINN 进行了深度重构，提出了一种**“基于物理法则引导的弱监督学习机制”**。

**核心技术亮点：**
1. **多任务“双核” AI 大脑 (`networks_wifi8`)：** 模型在提取 5 维物理特征（新增干扰感知维度）后，兵分两路。一头通过 Softmax 决定资源划分比例 $Q$，另一头通过 Sigmoid 独立预测各个切片的 DRU 激活概率。
2. **物理法则引导的无监督觉醒 (`custom_loss_wifi8`)：** 我们没有给 AI 提供关于 DRU 的“标准答案”。而是将通信底层的**导频开销代价**与**频率分集增益**抽象为数学公式嵌入损失函数。AI 在经历了排队延迟爆炸的“毒打”后，无师自通地学会了在强干扰下主动拉起 DRU 护盾。
3. **DRU 敏感性分析与博弈论涌现 (Sensitivity Analysis)：** 我们对 DRU 的物理开销（10%, 15%, 20%）进行了消融实验。实验证明，AI 具备了类似人类的“经济学博弈思维”：当切换代价高达 20% 时，AI 变得极为谨慎；当代价仅为 10% 时，AI 则果断开启护盾。这赋予了深度学习模型极强的物理可解释性。
4. **终极降维打击：** 在 64 个并发切片且遭遇强窄带干扰的极端测试中，传统算法不仅耗时数秒且延迟全部超标；而 Wi-Fi 8 专属 PINN 仅需 **1 毫秒** 即可完成自适应抗干扰调度，以极低的系统开销完美守住了 URLLC 的延迟生命线。

### 全局项目结构图 (更新至第二阶段)

```text
AQPS_PINN_Slicing/
│
├── checkpoints/                    
│   ├── best_aqps_pinn.pth          # (Phase 1) 理想信道最佳模型
│   └── phase2_uhr/                 # (Phase 2) Wi-Fi 8 专属权重库
│       ├── best_aqps_pinn_wifi8.pth
│       └── best_aqps_pinn_wifi8_dru10/15/20.pth # 消融实验权重
│
├── problem_descriptors/            
│   └── slicing_params.json         # 全局物理参数与 SLA 阈值配置
│
├── data_generation/                
│   ├── traffic_simulator.py        # (Phase 1) 4维理想特征生成器
│   ├── dataset_builder.py          # (Phase 1) 理想数据集印钞机
│   └── phase2_uhr/                 # (Phase 2) 突发干扰生成引擎
│       ├── traffic_simulator_wifi8.py  # 注入 50%~95% 容量毁坏的窄带干扰
│       └── dataset_builder_wifi8.py    # 生成包含 5 维特征的抗干扰数据集
│
├── dataset/                        
│   ├── train_data.pt / test_data.pt    # (Phase 1) 4维理想考卷
│   └── phase2_uhr/                     # (Phase 2) 5维抗干扰考卷
│       └── train_uhr.pt / test_uhr.pt  
│
├── non_linear_optimization/        
│   └── gekko_solver.py             # 传统基线：无法处理 DRU 的数学求解器
│
├── pinn_model/                     
│   ├── networks.py / custom_loss.py    # (Phase 1) 单头输出模型
│   └── phase2_uhr/                     # (Phase 2) Wi-Fi 8 跨层调度双核模型
│       ├── networks_wifi8.py           # 5维输入 + [Q, DRU] 双头输出架构
│       └── custom_loss_wifi8.py        # 引入 CRU/DRU 信道退化与频率分集惩罚机制
│
├── results/                        
│   ├── scalability_test.png            # (Phase 1) 纯耗时降维打击图
│   └── phase2_uhr/                     # (Phase 2) 诸神之战结果图
│       └── sensitivity_analysis.png    # 震撼审稿人的 DRU 敏感性与鲁棒性图表
│
├── train.py                        # (Phase 1) 训练脚本
├── train_wifi8.py                  # (Phase 2) Wi-Fi 8 专属训练脚本 (支持敏感性批量训练)
├── evaluate.py                     # (Phase 1) 评测脚本
├── evaluate_wifi8.py               # (Phase 2) 终极评测与画图脚本
│
├── requirements.txt                
└── README.md