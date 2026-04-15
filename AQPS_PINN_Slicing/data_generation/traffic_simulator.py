import json
import numpy as np
import os


class TrafficSimulator:
    def __init__(self, config_path="../problem_descriptors/slicing_params.json"):
        """
        初始化流量模拟器，读取全局 JSON 配置
        """
        # 确保路径正确
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置文件: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.sys_params = self.config["system_parameters"]
        self.slice_defs = self.config["slice_definitions"]
        self.traffic_sim = self.config["traffic_simulation"]

    def generate_dynamic_slices(self, num_slices):
        """
        核心函数：根据输入的切片数量(如 3, 8, 64)，动态生成网络状态。
        自动按比例分配切片类型 (例如: 15% URLLC, 35% eMBB, 50% BE)
        """
        # 1. 确定每种切片的数量
        num_urllc = max(1, int(num_slices * 0.15))
        num_embb = max(1, int(num_slices * 0.35))
        num_be = num_slices - num_urllc - num_embb

        slice_types = ['URLLC'] * num_urllc + ['eMBB'] * num_embb + ['BE'] * num_be

        # 初始化存储数组
        lambdas = np.zeros(num_slices)
        etas = np.zeros(num_slices)
        buffers = np.zeros(num_slices)
        psis = np.zeros(num_slices)
        w_qos = np.zeros(num_slices)
        b_max = np.zeros(num_slices)

        # 2. 为每一个动态生成的切片赋予随机流量和基础属性
        for i, s_type in enumerate(slice_types):
            # 获取该类型的基础限制 (从 JSON 读取)
            psis[i] = self.slice_defs["psi"][s_type]
            w_qos[i] = self.slice_defs["W_qos_ms"][s_type]
            b_max[i] = self.slice_defs["B_max_packets"][s_type]

            # 随机生成动态环境数据 (模拟真实基站拥塞波动)
            # 流量到达率 lambda
            lambdas[i] = np.random.uniform(self.traffic_sim["lambda_min"],
                                           self.traffic_sim["lambda_max"])
            # 频谱效率 eta (信道质量)
            etas[i] = np.random.uniform(self.traffic_sim["eta_min"],
                                        self.traffic_sim["eta_max"])
            # 当前缓冲区积压量 buffer (0 到 B_max 之间)
            buffers[i] = np.random.uniform(0, b_max[i])

        # 3. 极端拥塞缩放：如果总流量超过了基站物理极限 1720，我们要进行截断或缩放
        # 保证总流量在物理极限附近波动，制造 "Corner Cases" 供 AI 学习
        total_lambda = np.sum(lambdas)
        if total_lambda > self.sys_params["mu_max"] * 0.95:
            scale_factor = (self.sys_params["mu_max"] * 0.95) / total_lambda
            lambdas = lambdas * scale_factor

        return {
            "slice_types": slice_types,
            "lambdas": lambdas,  # 流量到达率
            "etas": etas,  # 频谱效率
            "buffers": buffers,  # 缓冲区当前状态
            "psis": psis,  # 基准优先级
            "W_qos": w_qos,  # 延迟阈值
            "B_max": b_max  # 缓冲区上限
        }


# ================= 测试代码 =================
if __name__ == "__main__":
    test_config_path = os.path.join(os.path.dirname(__file__), "../problem_descriptors/slicing_params.json")

    try:
        simulator = TrafficSimulator(test_config_path)
        test_counts = simulator.config["scalability_settings"]["test_slice_counts"]  # [3, 8, 16, 32, 64]

        print("开始动态生成不同规模的切片数据...\n")

        for count in test_counts:
            data = simulator.generate_dynamic_slices(count)
            print(f"=== 生成规模: {count} 个并发切片 ===")
            print(f"类型分布: {data['slice_types'].count('URLLC')} URLLC, "
                  f"{data['slice_types'].count('eMBB')} eMBB, "
                  f"{data['slice_types'].count('BE')} BE")
            print(f"总并发流量 (Σλ): {np.sum(data['lambdas']):.2f} packets/s (物理极限: 1720)")
            print(f"前3个切片的信噪比(η): {data['etas'][:3]}")
            print("-" * 50)

    except Exception as e:
        print(f"运行出错: {e}")
        print("请确保你在项目的 data_generation 目录下运行，并且上一步的 JSON 文件路径正确！")