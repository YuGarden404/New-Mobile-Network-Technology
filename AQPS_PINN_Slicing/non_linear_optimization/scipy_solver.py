import numpy as np
from scipy.optimize import minimize
import time
import json
import os


class MathSolver:
    def __init__(self, config_path="../problem_descriptors/slicing_params.json"):
        """
        传统数学运筹学优化器 (模拟论文1的 IPOPT/Z3 行为)
        绝对安全但过于缓慢的基准模型，为 AI 生成标准答案
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件路径错误: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.mu_max = self.config["system_parameters"]["mu_max"]

    def _compute_delay_ms(self, Q, lambdas):
        """
        排队论公式：计算给定分配比例 Q 下的平均排队延迟 W (ms)
        采用 M/M/1 模型: W = (ρ^2) / (λ * (1-ρ))
        """
        mu = Q * self.mu_max
        rho = lambdas / (mu + 1e-9)
        delay = np.zeros_like(Q)
        for i in range(len(Q)):
            if rho[i] >= 0.95:
                rho_safe = 0.95
                base_delay = (rho_safe ** 2) / (lambdas[i] * (1 - rho_safe) + 1e-9) * 1000.0
                penalty_gradient = 50000.0 * (rho[i] - 0.95)
                delay[i] = base_delay + penalty_gradient
            else:
                delay[i] = (rho[i] ** 2) / (lambdas[i] * (1 - rho[i]) + 1e-9) * 1000.0
        return delay

    def solve(self, traffic_data, initial_Q=None):
        """
        执行非线性优化求解 (QCQP)
        """
        lambdas = traffic_data['lambdas']
        psis = traffic_data['psis']
        etas = traffic_data['etas']
        n_slices = len(lambdas)

        """
        为了解决传统 SLSQP 优化器在极端流量拥塞下因可行域消失而导致求解崩溃的问题，
        本文在基准算法中引入了基于 L1 精确惩罚函数法（L1 Exact Penalty Method [Nocedal & Wright, 2006]）的软约束机制
        我们将严格的 QoS 延迟约束转化为目标函数中的违约惩罚项，
        并结合不同切片的 SLA 优先级权重（Shadow Prices [Kelly et al., 1998]）构建惩罚梯度
        惩罚因子 mu 被设定为 10^4，以确保其在数值上严格大于最优拉格朗日乘子的上界，
        从而在避免 Hessian 矩阵病态的同时，保证了拥塞状态下高优先级 URLLC 切片的绝对降级保障
        
        Kelly, F. P., Maulloo, A. K., & Tan, D. K. (1998). Rate control for communication networks: shadow prices, proportional fairness and stability. Journal of the Operational Research Society
        """
        def objective(Q):
            # 目标：最小化“加权总时延”，越紧急的业务 psi 越大，要求时延越小
            delays = self._compute_delay_ms(Q, lambdas)
            # 基础目标：最小化加权总时延
            base_cost = np.sum(psis * delays)
            # np.maximum(0, ...) 意味着没超标就不罚款，超标了就按超出的毫秒数重罚
            qos_violations = np.maximum(0, delays - etas)
            # 乘以 10000.0 的超大权重，强迫优化器在“必死局”中也要拼命压低违约程度
            penalty = np.sum(10000.0 * psis * qos_violations)
            return base_cost + penalty

        # 约束一: 所有切片的资源比例总和必须等于 1
        def constraint_sum(Q):
            return np.sum(Q) - 1.0

        # 约束二: 分配给每个切片的处理能力 mu 必须大于其流量 lambda
        def constraint_capacity(Q):
            return (Q * self.mu_max) - lambdas - 1e-3

        bounds = [(1e-5, 0.999) for _ in range(n_slices)]
        constraints = [
            {'type': 'eq', 'fun': constraint_sum},
            {'type': 'ineq', 'fun': constraint_capacity}
        ]

        min_Q_needed = (lambdas + 0.1) / self.mu_max
        remaining_Q = 1.0 - np.sum(min_Q_needed)

        if remaining_Q <= 0:
            initial_Q = lambdas / np.sum(lambdas)
        else:
            weights = np.sqrt(psis * lambdas)
            initial_Q = min_Q_needed + remaining_Q * (weights / np.sum(weights))

        start_time = time.perf_counter()

        # 调用 SciPy 内部的序列二次规划(SLSQP)求解器
        res = minimize(
            objective,
            initial_Q,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 10000, 'ftol': 1e-6},
        )

        cost_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            'Q_opt': res.x if res.success else initial_Q,   # 返回最优资源分配比例或兜底解
            'success': res.success,                         # 是否成功找到最优解
            'solve_time_ms': cost_time_ms,                  # 求解耗时
            'iterations': getattr(res, 'nit', 0)            # 迭代次数
        }


# ================= 测试代码 =================
if __name__ == "__main__":
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
    from AQPS_PINN_Slicing.data_generation.traffic_simulator import TrafficSimulator

    config_path = os.path.join(os.path.dirname(__file__), "../problem_descriptors/slicing_params.json")

    print("[测试] 启动数学运筹学优化器引擎...")
    simulator = TrafficSimulator(config_path)
    solver = MathSolver(config_path)

    # 模拟生成一组 8 个切片的数据
    traffic_data = simulator.generate_dynamic_slices(8)
    print(f"\n[考卷] 生成 8 个切片，总流量 Σλ = {np.sum(traffic_data['lambdas']):.2f} / 1720")

    # 让数学优化器做题
    print("\n[解题] 传统数学优化器(IPOPT/SLSQP)正在暴力搜索最优解...")
    result = solver.solve(traffic_data)

    if result['success']:
        print(f"\n✅ [求解成功] 优化器成功收敛！")
        print(f"最优资源分配比例 (Q_opt): {np.round(result['Q_opt'], 3)}")
    else:
        print(f"\n❌ [优化崩溃] 陷入梯度断层或无可行解！")
        print(f"⚠️ 触发基站降级机制，输出兜底按比例分配结果: {np.round(result['Q_opt'], 3)}")
    print(f"比例总和: {np.sum(result['Q_opt']):.2f} (必须等于 1.0)")
    print(f"迭代次数: {result['iterations']} 步")
    print(f"求解耗时: {result['solve_time_ms']:.2f} ms")