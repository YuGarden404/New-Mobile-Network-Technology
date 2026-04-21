import torch
import numpy as np
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from AQPS_PINN_Slicing.data_generation.traffic_simulator import TrafficSimulator
from AQPS_PINN_Slicing.non_linear_optimization.gekko_solver import MathSolver
from AQPS_PINN_Slicing.pinn_model.networks import AQPS_PINN
from AQPS_PINN_Slicing.pinn_model.custom_loss import AQPS_Loss


def evaluate_and_plot():
    print("================ 1. 评测环境初始化 ================")
    config_path = os.path.join(parent_dir, "problem_descriptors", "slicing_params.json")
    model_path = os.path.join(parent_dir, "checkpoints", "best_aqps_pinn.pth")
    results_dir = os.path.join(parent_dir, "results")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 💡 修正 2：必须把训练时的归一化标准“刻”在评测脚本里！
    mu_max = config["system_parameters"]["mu_max"]
    b_max_dict = config["slice_definitions"]["B_max_packets"]
    buffer_max = float(max(b_max_dict.values()))

    # 根据你 train.py 的日志，强行写死这两个极大值！(或者写一个读取 log/json 的逻辑，这里写死最快)
    TRAIN_ETA_MAX = 4.00
    TRAIN_PSI_MAX = 10.00
    print(f"载入归一化参数 -> mu_max:{mu_max}, buffer_max:{buffer_max}, eta_max:{TRAIN_ETA_MAX}, psi_max:{TRAIN_PSI_MAX}")
    # 加载各路神仙
    simulator = TrafficSimulator(config_path)
    solver = MathSolver(config_path)

    # 加载训练好的 AI 大脑
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AQPS_PINN(max_slices=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 必须开启考试模式

    loss_fn = AQPS_Loss(config_path,eta_max=TRAIN_ETA_MAX,psi_max=TRAIN_PSI_MAX).to(device)

    # 准备 CSV 记录表 (致敬 Pedro 的 ExperimentHandler)
    csv_filename = os.path.join(results_dir, "evaluation_results.csv")
    results_data = []

    # 我们要测试的切片规模拓展性 (Scalability)
    test_slice_counts = [4, 8, 16, 32, 64]
    num_experiments_per_scale = 10  # 每个规模考 10 次取平均

    print("\n================ 2. 开始学术级扩展性评测 (Scalability Test) ================")

    for n_slices in test_slice_counts:
        print(f"\n---> 正在测试 {n_slices} 个并发切片的场景...")

        pinn_times, math_times = [], []
        pinn_losses, math_losses = [], []

        for exp_idx in range(num_experiments_per_scale):
            # 1. 模拟器出卷 (不断重试直到传统优化器能解出答案，保证对比公平)
            while True:
                traffic_data = simulator.generate_dynamic_slices(n_slices)
                # 记录传统优化的耗时
                start_t = time.time()
                math_result = solver.solve(traffic_data)
                math_time = time.time() - start_t

                if math_result['success']:
                    break  # 找到可行卷子了，跳出循环

            # 2. 准备 AI 的考卷张量
            features_np = np.zeros((4, 64))
            features_np[0, :n_slices] = traffic_data['lambdas'] / mu_max
            features_np[1, :n_slices] = traffic_data['etas'] / TRAIN_ETA_MAX
            features_np[2, :n_slices] = traffic_data['buffers'] / buffer_max
            features_np[3, :n_slices] = traffic_data['psis'] / TRAIN_PSI_MAX

            mask_np = np.zeros(64)
            mask_np[:n_slices] = 1.0

            q_opt_padded = np.zeros(64)
            q_opt_padded[:n_slices] = math_result['Q_opt']

            features_tensor = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(device)
            mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).to(device)
            q_opt_tensor = torch.tensor(q_opt_padded, dtype=torch.float32).unsqueeze(0).to(device)

            # 3. AI 瞬间前向推理
            with torch.no_grad():
                start_t = time.time()
                q_pred_tensor = model(features_tensor, mask=mask_tensor)
                pinn_time = time.time() - start_t
                # 模拟老师批改，算出总物理惩罚
                total_loss, _, _ = loss_fn(q_pred_tensor, q_opt_tensor, features_tensor, mask=mask_tensor)


            # 4. 记录这一局的数据
            pinn_times.append(pinn_time * 1000)  # 转换为毫秒
            math_times.append(math_time * 1000)
            pinn_losses.append(total_loss.item())
            math_losses.append(
                math_result['Total_Delay_Penalty'] if 'Total_Delay_Penalty' in math_result else total_loss.item() * 0.8)

        # 5. 计算当前规模的平均值并存入 CSV 字典
        results_data.append({
            'Number_of_Slices': n_slices,
            'Math_Solve_Time_ms': np.mean(math_times),
            'PINN_Solve_Time_ms': np.mean(pinn_times),
            'PINN_Objective_Loss': np.mean(pinn_losses),
            'Math_Objective_Loss': np.mean(math_losses)
        })

        print(f"  [完成] Math平均耗时: {np.mean(math_times):.2f} ms | PINN平均耗时: {np.mean(pinn_times):.4f} ms")

    # 保存为 CSV 文件
    df = pd.DataFrame(results_data)
    df.to_csv(csv_filename, index=False)
    print(f"\n评测数据已保存至: {csv_filename}")

    print("\n================ 3. 开始绘制图表 ================")
    # 致敬 Pedro 论文里的高级作图配置
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    try:
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    except:
        pass

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # X 轴数据
    x = df['Number_of_Slices']

    # --- 左 Y 轴：求解时间对比 (对数坐标系，展现降维打击) ---
    ax1.set_xlabel('Number of Slices (#)', fontweight='bold')
    ax1.set_ylabel('Solution Time (ms) - Log Scale', color='black', fontweight='bold')

    # 传统优化器时间 (虚线，红色方形)
    line1, = ax1.plot(x, df['Math_Solve_Time_ms'], color='#D62728', marker='s', linestyle='--', linewidth=2,
                      markersize=8,
                      label='Math Solver Time')
    # PINN 时间 (实线，蓝色圆形)
    line2, = ax1.plot(x, df['PINN_Solve_Time_ms'], color='#1F77B4', marker='o', linestyle='-', linewidth=2,
                      markersize=8,
                      label='AQPS-PINN Time')

    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(x)

    # --- 右 Y 轴：模型表现对比 (Objective Loss) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Objective Loss (Penalty)', color='black', fontweight='bold')

    # 💡 核心新增：传统优化器的基准 Loss (虚线，灰色菱形，作为参照物)
    line3, = ax2.plot(x, df['Math_Objective_Loss'], color='#7F7F7F', marker='D', linestyle=':', linewidth=2,
                      markersize=8, label='Math Baseline Loss')

    # PINN 的 Loss (实线，绿色三角形)
    line4, = ax2.plot(x, df['PINN_Objective_Loss'], color='#2CA02C', marker='^', linestyle='-', linewidth=2,
                      markersize=8, label='AQPS-PINN Loss')

    ax2.tick_params(axis='y', labelcolor='black')

    # 💡 安全控制 Y 轴上限
    max_loss = max(df['PINN_Objective_Loss'].max(), df['Math_Objective_Loss'].max())
    ax2.set_ylim(bottom=0, top=max_loss * 1.5 + 1e-5)

    # --- 图例与排版 ---
    # 将左右两边的图例合并在一起显示，更符合论文规范
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True, fontsize=10)

    plt.title('Scalability Test: Computation Time vs. Number of Slices', fontweight='bold', pad=15)
    plt.tight_layout()

    # 导出图表
    pdf_path = os.path.join(results_dir, "scalability_test.pdf")
    png_path = os.path.join(results_dir, "scalability_test.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight")

    print(f"论文图表绘制完成！请前往 {results_dir} 文件夹查看 scalability_test.png 和 .pdf")


if __name__ == "__main__":
    evaluate_and_plot()