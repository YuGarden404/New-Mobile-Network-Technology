# 项目文档

## 项目结构图

```
AQPS_PINN_Slicing/
│
├── problem_descriptors/            # [对标论文图] 存放所有超参数和网络配置
│   └── slicing_params.json         # 全局物理参数、SLA阈值、优先级的 JSON 配置文件
│
├── data_generation/                # [新增] 数据引擎：因为我们要训练 AI，必须有数据
│   ├── traffic_simulator.py        # 模拟生成真实的泊松流量、突发信道衰落和缓冲区积压
│   └── dataset_builder.py          # 离线调用优化器，生成供 AI 学习的 (X, Y_opt) 数据集
│
├── non_linear_optimization/        # [对标论文图] 传统数学求解器引擎
│   ├── scipy_solver.py             # 使用 SciPy SLSQP 实现的 QCQP 求解器 (替代较慢的Z3)
│   └── optimizer_engine.py         # 封装冷启动(Baseline)与热启动(Ours)的调用接口
│
├── pinn_model/                     # [核心创新] 物理信息神经网络引擎
│   ├── networks.py                 # 多任务 DNN 架构 (带 Softmax 可行解映射层)
│   └── custom_loss.py              # 最硬核的代码：可微 M/M/1 排队层与动态紧迫性 Loss
│
├── handlers/                       # [对标论文图] 数据加载与流转工具
│   └── data_loader.py              # PyTorch Dataset/DataLoader 的封装
│
├── graphs/                         # [对标论文图] 实验结果可视化脚本
│   ├── plot_convergence.py         # 绘制 AI 训练时的 Loss 收敛曲线
│   ├── plot_runtime.py             # 绘制 纯数学 vs 纯AI vs 热启动 的毫秒级耗时对比
│   └── plot_sla_violations.py      # 绘制 在信道恶化时的 SLA 违约率对比图
│
├── train.py                        # 执行离线训练的入口脚本
├── evaluate.py                     # 执行在线实时调度与对比测试的入口脚本
├── requirements.txt                # 运行环境依赖 (torch, scipy, matplotlib等)
└── README.md                       # 项目文档
```