import os
import numpy as np
import torch
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from AQPS_PINN_Slicing.data_generation.traffic_simulator import TrafficSimulator
from AQPS_PINN_Slicing.non_linear_optimization.scipy_solver import MathSolver


class DatasetBuilder:
    def __init__(self, config_path="../problem_descriptors/slicing_params.json"):
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
        print(f"开始构建高质量训练集 (目标样本数: {total_samples})...")
        print("注: 如果传统优化器在极端数据上崩溃，系统将自动丢弃该样本以保证数据纯净。\n")

        X_data = []
        Y_data = []

        failed_count = 0

        pbar = tqdm(total=total_samples, desc="Generating")

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
                pbar.update(1)
            else:
                failed_count += 1

        pbar.close()

        X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32)

        save_path = os.path.join(self.save_dir, filename)
        torch.save({'X': X_tensor, 'Y': Y_tensor}, save_path)

        print(f"\n数据集构建完成！")
        print(f"成功保存 {total_samples} 条纯净数据至: {save_path}")
        print(f"过滤掉由于数学崩溃导致的废弃样本数: {failed_count} 条")


# ================= 运行构建 =================
if __name__ == "__main__":
    builder = DatasetBuilder()
    # 生成训练集
    builder.build_dataset(total_samples=1000, slice_counts=[3, 8, 16, 32], filename="train_data.pt")
    # 生成测试集
    builder.build_dataset(total_samples=200, slice_counts=[3, 8, 16, 32, 64], filename="test_data.pt")