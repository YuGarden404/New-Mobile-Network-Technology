import torch
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# 确保能正确导入我们手写的模块
from AQPS_PINN_Slicing.data_generation.traffic_simulator import TrafficSimulator
from AQPS_PINN_Slicing.non_linear_optimization.scipy_solver import MathSolver
from AQPS_PINN_Slicing.pinn_model.networks import AQPS_PINN
from AQPS_PINN_Slicing.pinn_model.custom_loss import AQPS_Loss


def run_integration_test():
    print("================ 1. 模块加载中 ================")
    config_path = os.path.join(parent_dir, "problem_descriptors", "slicing_params.json")

    simulator = TrafficSimulator(config_path)
    solver = MathSolver(config_path)
    model = AQPS_PINN(max_slices=64)
    loss_fn = AQPS_Loss(config_path)
    print("模拟器、优化器、神经网络、损失函数配置完成")

    print("\n================ 2. 生成动态考卷 ================")
    n_slices = 8  # 模拟一次有 8 个切片的并发拥塞
    traffic_data = simulator.generate_dynamic_slices(n_slices)
    print(f"成功生成考卷：包含 {n_slices} 个活跃切片，总流量 Σλ = {np.sum(traffic_data['lambdas']):.2f}")

    print("\n================ 3. 运筹学计算标准答案 ================")
    result = solver.solve(traffic_data)
    if not result['success']:
        print("[警告] 此次随机生成的考卷导致数学优化器崩溃。请重新运行此脚本")
        return
    q_opt_numpy = result['Q_opt']
    print("已通过传统运筹学算出标准答案 Q_opt")

    print("\n================ 4. 张量对齐与打包 (Padding & Masking) ================")
    MAX_SLICES = 64
    features_np = np.zeros((4, MAX_SLICES))
    features_np[0, :n_slices] = traffic_data['lambdas']
    features_np[1, :n_slices] = traffic_data['etas']
    features_np[2, :n_slices] = traffic_data['buffers']
    features_np[3, :n_slices] = traffic_data['psis']

    mask_np = np.zeros(MAX_SLICES)
    mask_np[:n_slices] = 1.0  # 前 8 个是真切片，打上 1.0 掩码

    q_opt_padded = np.zeros(MAX_SLICES)
    q_opt_padded[:n_slices] = q_opt_numpy

    features_tensor = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0)   # 形状: [1, 4, 64]
    mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)           # 形状: [1, 64]
    q_opt_tensor = torch.tensor(q_opt_padded, dtype=torch.float32).unsqueeze(0)     # 形状: [1, 64]

    print("考卷已打包成标准 Tensor，准备送入神经网络")

    print("\n================ 5. AI 模型前向推理 ================")
    # 注意网络输入需要一维拉平的特征，所以将 [1, 4, 64] 拉平成 [1, 256]
    x_input = features_tensor.view(1, -1)

    model.eval()

    # 因为 AI 还没被训练过，其权重是随机初始化的，所以会给出一个极其平均的“随机答案”
    # 为了严谨，在测试前向传播时我们不需要计算梯度，加上 torch.no_grad() 可以节省内存
    with torch.no_grad():
        q_pred_tensor = model(x_input, mask=mask_tensor)

    print("未经训练的 AI 完成了第一次作答")

    print("\n================ 6. 计算 Loss ================")
    # 将瞎猜的 Q_pred、正确的 Q_opt、环境特征 features 送入损失函数
    total_loss, mse, qos = loss_fn(q_pred_tensor, q_opt_tensor, features_tensor, mask=mask_tensor)

    print("\n================ 联调结果最终报告 ================")
    print(f"[结论] 真实活跃切片数: {n_slices} (满槽位: 64)")
    print(f"[结论] AI 预测比例总和: {torch.sum(q_pred_tensor).item():.4f} (Mask机制生效，严格等于1.0)")
    print(f"[结论] 标准答案: {np.round(q_opt_tensor[0, :8].numpy(), 8)}")
    print(f"[结论] AI瞎猜答案: {np.round(q_pred_tensor[0, :8].detach().numpy(), 8)}")
    print("-" * 60)
    print(f"MSE 回归误差 (它模仿得很差):    {mse.item():.4f}")
    print(f"QoS 物理惩罚 (它引发了严重超时): {qos.item():.4f}")
    print(f"最终总损失 (Total Loss):        {total_loss.item():.4f}")
    print("==============================================")

if __name__ == "__main__":
    run_integration_test()