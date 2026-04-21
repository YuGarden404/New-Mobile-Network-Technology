import os
import numpy as np
import torch
import sys
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from AQPS_PINN_Slicing.data_generation.traffic_simulator import TrafficSimulator
from AQPS_PINN_Slicing.non_linear_optimization.gekko_solver import MathSolver


class DatasetBuilder:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(parent_dir, "problem_descriptors", "slicing_params.json")

        self.simulator = TrafficSimulator(config_path)
        self.solver = MathSolver(config_path)
        self.save_dir = "../dataset"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def build_dataset(self, total_samples=2000, slice_counts=[3, 8, 16, 32], filename="train_dataset.pt"):
        """
        核心数据生成流水线：
        让模拟器不断出题，让求解器不断做题。只保留做题成功的 (X, Y) 数据对。
        """
        print(f"\n启动 Gekko 运算引擎...")
        print(f"开始构建高质量数据集 (目标样本数: {total_samples})...")
        print("注: 极度拥塞导致无解的脏数据将被自动拦截，确保 Ground Truth 的绝对纯净。\n")

        X_data = []
        Y_data = []

        failed_count = 0
        total_constraints_removed = 0

        pbar = tqdm(total=total_samples, desc=f"Generating {filename}")

        while len(X_data) < total_samples:
            n_slices = np.random.choice(slice_counts)
            traffic_data = self.simulator.generate_dynamic_slices(n_slices)
            result = self.solver.solve(traffic_data)

            if result['success']:
                MAX_SLICES = 64

                features = np.zeros((4, MAX_SLICES))  # 4个特征: lambda, eta, buffer, psi
                labels = np.zeros(MAX_SLICES)

                features[0, :n_slices] = traffic_data['lambdas']
                features[1, :n_slices] = traffic_data['etas']
                features[2, :n_slices] = traffic_data['buffers']
                features[3, :n_slices] = traffic_data['psis']
                labels[:n_slices] = result['Q_opt']

                X_data.append(features.flatten())
                Y_data.append(labels)
                # 记录 Gekko 妥协了多少次约束
                total_constraints_removed += result['constraints_removed']
                pbar.update(1)
            else:
                failed_count += 1

        pbar.close()

        X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32)

        save_path = os.path.join(self.save_dir, filename)
        torch.save({'X': X_tensor, 'Y': Y_tensor}, save_path)

        print(f"\n数据集 [{filename}] 构建完成")
        print(f"成功保存 {total_samples} 条纯净数据至: {save_path}")
        print(f"过滤掉的数学崩溃样本数: {failed_count} 条")
        print(f"生成本数据集共被迫删除了 {total_constraints_removed} 个 QoS 约束\n")

# ================= 运行构建 =================
if __name__ == "__main__":
    builder = DatasetBuilder()
    # 生成训练集
    builder.build_dataset(total_samples=10000, slice_counts=[3, 8, 16, 32], filename="train_data.pt")
    # 生成测试集
    builder.build_dataset(total_samples=2000, slice_counts=[3, 8, 16, 32, 64], filename="test_data.pt")