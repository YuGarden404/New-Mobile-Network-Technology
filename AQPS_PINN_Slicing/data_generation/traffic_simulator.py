import json
import numpy as np
import os


class TrafficSimulator:
    def __init__(self, config_path="../problem_descriptors/slicing_params.json"):
        """
        初始化流量模拟器，读取全局 JSON 配置
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件路径错误: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.sys_params = self.config["system_parameters"]
        self.slice_defs = self.config["slice_definitions"]
        self.traffic_sim = self.config["traffic_simulation"]

    def generate_dynamic_slices(self, n_slices):
        """
        核心函数：根据输入的切片数量(如 3, 8, 16, 32, 64)，动态生成网络状态
        自动按比例分配切片类型 (默认：15% URLLC, 35% eMBB, 50% BE)
        """
        num_urllc = max(1, int(n_slices * 0.15))
        num_embb = max(1, int(n_slices * 0.35))
        num_be = n_slices - num_urllc - num_embb
        slice_types = ['URLLC'] * num_urllc + ['eMBB'] * num_embb + ['BE'] * num_be

        lambdas = np.zeros(n_slices)
        etas = np.zeros(n_slices)
        buffers = np.zeros(n_slices)
        psis = np.zeros(n_slices)
        w_qos = np.zeros(n_slices)
        b_max = np.zeros(n_slices)

        for i, slice_type in enumerate(slice_types):
            psis[i] = self.slice_defs["psi"][slice_type]
            w_qos[i] = self.slice_defs["W_qos_ms"][slice_type]
            b_max[i] = self.slice_defs["B_max_packets"][slice_type]
            lambdas[i] = np.random.uniform(self.traffic_sim["lambda_min"], self.traffic_sim["lambda_max"])
            etas[i] = np.random.uniform(self.traffic_sim["eta_min"], self.traffic_sim["eta_max"])
            buffers[i] = np.random.uniform(0, b_max[i])

        total_lambda = np.sum(lambdas)
        if total_lambda > self.sys_params["mu_max"] * 0.95:
            scale_factor = (self.sys_params["mu_max"] * 0.95) / total_lambda
            lambdas = lambdas * scale_factor

        return {
            "slice_types": slice_types, # 切片类型
            "lambdas": lambdas,         # 流量到达率
            "etas": etas,               # 频谱效率
            "buffers": buffers,         # 缓冲区当前状态
            "psis": psis,               # 基准优先级
            "W_qos": w_qos,             # 延迟阈值
            "B_max": b_max              # 缓冲区上限
        }


# ================= 测试代码 =================
if __name__ == "__main__":
    test_config_path = os.path.join(os.path.dirname(__file__), "../problem_descriptors/slicing_params.json")

    try:
        simulator = TrafficSimulator(test_config_path)
        test_counts = simulator.config["scalability_settings"]["test_slice_counts"]  # [3, 8, 16, 32, 64]

        print("[测试] 动态生成不同规模的切片数据...\n")
        print("-" * 60)

        for count in test_counts:
            data = simulator.generate_dynamic_slices(count)
            print(f"[测试] === 生成规模: {count} 个切片 ===")
            print(f"[测试] 类型分布: {data['slice_types'].count('URLLC')} URLLC, "
                  f"{data['slice_types'].count('eMBB')} eMBB, "
                  f"{data['slice_types'].count('BE')} BE")
            print(f"[测试] 总并发流量 (Σλ): {np.sum(data['lambdas']):.2f} packets/s (物理极限: 1720)")
            print(f"[测试] 前3个切片的信噪比(η): {data['etas'][:3]}")
            print("-" * 60)

    except Exception as e:
        print(f"[测试] 运行出错: {e}")