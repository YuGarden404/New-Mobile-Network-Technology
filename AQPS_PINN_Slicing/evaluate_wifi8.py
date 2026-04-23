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

from data_generation.phase2_uhr.traffic_simulator_wifi8 import TrafficSimulatorWiFi8
from non_linear_optimization.gekko_solver import MathSolver
from pinn_model.phase2_uhr.networks_wifi8 import AQPS_PINN_WiFi8
from pinn_model.phase2_uhr.custom_loss_wifi8 import AQPS_Loss_WiFi8


def evaluate_sensitivity():
    print("================ 多模型评测环境初始化 ================")
    config_path = os.path.join(current_dir, "problem_descriptors", "slicing_params.json")
    results_dir = os.path.join(current_dir, "results", "phase2_uhr")
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    mu_max = config["system_parameters"]["mu_max"]
    b_max_dict = config["slice_definitions"]["B_max_packets"]
    buffer_max = float(max(b_max_dict.values()))
    TRAIN_ETA_MAX = 4.00
    TRAIN_PSI_MAX = 10.00

    simulator = TrafficSimulatorWiFi8(config_path)
    solver = MathSolver(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 3 个不同性格的 AI
    dru_configs = [0.10, 0.15, 0.20]
    models = {}
    loss_fns = {}

    for dl in dru_configs:
        model_name = f"best_aqps_pinn_wifi8_dru{int(dl * 100)}.pth"
        model_path = os.path.join(current_dir, "checkpoints", "phase2_uhr", model_name)

        m = AQPS_PINN_WiFi8(max_slices=64, input_features=5).to(device)
        m.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        m.eval()
        models[dl] = m

        lf = AQPS_Loss_WiFi8(config_path, eta_max=TRAIN_ETA_MAX, psi_max=TRAIN_PSI_MAX, dru_efficiency_loss=dl).to(
            device)
        lf.alpha_qos = 0.005
        loss_fns[dl] = lf

    results_data = []
    test_slice_counts = [4, 8, 16, 32, 64]
    num_experiments = 10

    print("\n================ 敏感性生存测试开始 ================")

    for n_slices in test_slice_counts:
        print(f"\n---> 测试 {n_slices} 并发切片...")

        math_times = []
        math_losses = []

        # 记录 3 个 AI 的表现
        pinn_losses = {0.10: [], 0.15: [], 0.20: []}
        dru_rates = {0.10: [], 0.15: [], 0.20: []}

        for exp_idx in range(num_experiments):
            while True:
                traffic_data = simulator.generate_dynamic_slices(n_slices)
                start_t = time.time()
                math_result = solver.solve(traffic_data)
                math_times.append(time.time() - start_t)
                if math_result['success']: break

            features_np = np.zeros((5, 64))
            features_np[0, :n_slices] = traffic_data['lambdas'] / mu_max
            features_np[1, :n_slices] = traffic_data['etas'] / TRAIN_ETA_MAX
            features_np[2, :n_slices] = traffic_data['buffers'] / buffer_max
            features_np[3, :n_slices] = traffic_data['psis'] / TRAIN_PSI_MAX
            features_np[4, :n_slices] = traffic_data['interferences']

            mask_np = np.zeros(64)
            mask_np[:n_slices] = 1.0
            q_opt_padded = np.zeros(64)
            q_opt_padded[:n_slices] = math_result['Q_opt']

            features_tensor = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(device)
            mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).to(device)
            q_opt_tensor = torch.tensor(q_opt_padded, dtype=torch.float32).unsqueeze(0).to(device)

            # Math 表现 (无法使用 DRU)
            math_dru_prob = torch.zeros((1, 64)).to(device)
            with torch.no_grad():
                # 借用 loss_fn 计算惩罚，dru_prob 是 0，开销参数不起作用
                mloss, _, _ = loss_fns[0.10](q_opt_tensor, math_dru_prob, q_opt_tensor, features_tensor,
                                             mask=mask_tensor)
                math_losses.append(mloss.item())

            # 评估 3 个不同性格的 AI
            for dl in dru_configs:
                with torch.no_grad():
                    q_pred, dru_prob = models[dl](features_tensor, mask=mask_tensor)
                    ploss, _, _ = loss_fns[dl](q_pred, dru_prob, q_opt_tensor, features_tensor, mask=mask_tensor)

                    pinn_losses[dl].append(ploss.item())
                    rate = (dru_prob[0, :n_slices] > 0.5).sum().item() / n_slices
                    dru_rates[dl].append(rate)

        # 记录均值
        row = {
            'Number_of_Slices': n_slices,
            'Math_Solve_Time_ms': np.mean(math_times) * 1000,
            'Math_UHR_Loss': np.mean(math_losses),
        }
        for dl in dru_configs:
            row[f'PINN_{int(dl * 100)}_Loss'] = np.mean(pinn_losses[dl])
            row[f'PINN_{int(dl * 100)}_DRU_Rate'] = np.mean(dru_rates[dl])

        results_data.append(row)
        print(
            f"[完成] Math Loss: {np.mean(math_losses):.1f} | PINN-10%: {np.mean(pinn_losses[0.10]):.1f} ({np.mean(dru_rates[0.10]) * 100:.1f}% DRU) | PINN-20%: {np.mean(pinn_losses[0.20]):.1f} ({np.mean(dru_rates[0.20]) * 100:.1f}% DRU)")

    df = pd.DataFrame(results_data)
    csv_path = os.path.join(results_dir, "sensitivity_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n================ 绘制 4 线对比图表 ================")
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    x = df['Number_of_Slices']

    ax1.set_xlabel('Number of Slices (#)', fontweight='bold')
    ax1.set_ylabel('UHR Physical Penalty (Log Scale)', color='black', fontweight='bold')

    # Math Loss (红色虚线)
    line_m, = ax1.plot(x, df['Math_UHR_Loss'], color='#D62728', marker='s', linestyle='--', linewidth=3, markersize=8,
                       label='Math Solver (No DRU)')
    # PINN 10% (绿色实线，最激进开启DRU，Loss极低)
    line_10, = ax1.plot(x, df['PINN_10_Loss'], color='#2CA02C', marker='o', linestyle='-', linewidth=2, markersize=8,
                        label='PINN (10% DRU Cost)')
    # PINN 15% (蓝色实线)
    line_15, = ax1.plot(x, df['PINN_15_Loss'], color='#1F77B4', marker='^', linestyle='-', linewidth=2, markersize=8,
                        label='PINN (15% DRU Cost)')
    # PINN 20% (紫色实线，最保守开启DRU，Loss略高)
    line_20, = ax1.plot(x, df['PINN_20_Loss'], color='#9467BD', marker='D', linestyle='-', linewidth=2, markersize=8,
                        label='PINN (20% DRU Cost)')

    ax1.set_yscale('log')
    ax1.set_xticks(x)

    # 图例
    lines = [line_m, line_10, line_15, line_20]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True, fontsize=11)

    plt.title('Sensitivity Analysis: Impact of DRU Physical Overhead on Robustness', fontweight='bold', pad=15)
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, "sensitivity_analysis.pdf"), format="pdf")
    plt.savefig(os.path.join(results_dir, "sensitivity_analysis.png"), format="png")
    print(f"\n图表已生成：{results_dir}/sensitivity_analysis.png")


if __name__ == "__main__":
    evaluate_sensitivity()