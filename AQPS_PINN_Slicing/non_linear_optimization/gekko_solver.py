#!/usr/bin/env python
# __author__ = "Pedro Heleno Isolani"
# __copyright__ = "Copyright 2019, QoS-aware WiFi Slicing"
# __license__ = "GPL"
# __version__ = "1.0"
# __maintainer__ = "Pedro Heleno Isolani"
# __email__ = "pedro.isolani@uantwerpen.be"
# __status__ = "Prototype"
#
# ''' Python class for solving IEEE 802.11 network slicing problem using gekko non-linear solvers'''

# 感谢 Pedro 的开源项目

from gekko import GEKKO
import numpy as np
import time
import os
import json

import random as r

def compute_progressive_lambdas(num_slices=0, rb_max_dequeuing_rate=None, lambda_factor=0.01, lambda_gap=0.1):
    if rb_max_dequeuing_rate is not None:
        new_lambdas = []
        crr_lambda = 0
        gap_dequeuing_rate = rb_max_dequeuing_rate * lambda_gap
        for i in range(num_slices):
            if i == (num_slices - 1):
                # Last one
                new_lambdas.append(round(rb_max_dequeuing_rate - gap_dequeuing_rate - sum(new_lambdas), 2))
            else:
                crr_lambda += lambda_factor
                new_lambdas.append(round(crr_lambda, 2))
    print_new_lambdas(new_lambdas)
    return new_lambdas


def compute_equal_lambdas(num_slices=0, rb_max_dequeuing_rate=None, lambda_gap=0.1):
    if rb_max_dequeuing_rate is not None:
        new_lambdas = []
        gap_dequeuing_rate = rb_max_dequeuing_rate * lambda_gap
        new_lambda = (rb_max_dequeuing_rate - gap_dequeuing_rate) / num_slices
        for i in range(num_slices):
            new_lambdas.append(new_lambda)
    print_new_lambdas(new_lambdas)
    return new_lambdas

def print_new_lambdas(new_lambdas):
    print('New lambdas: ', new_lambdas)
    print('New lambdas SUM: ', sum(new_lambdas))

def compute_random_lambdas(rb_max_dequeuing_rate=None, num_slices=0, lambda_gap=0.1):
    if rb_max_dequeuing_rate is not None:
        gap_dequeuing_rate = rb_max_dequeuing_rate * lambda_gap
        rb_max_dequeuing_rate = int(rb_max_dequeuing_rate - gap_dequeuing_rate)
        num_slices = (num_slices or r.randint(2, rb_max_dequeuing_rate)) - 1
        a = r.sample(range(1, rb_max_dequeuing_rate), num_slices) + [0, rb_max_dequeuing_rate]
        list.sort(a)
        new_lambdas = [a[i+1] - a[i] for i in range(len(a) - 1)]
        print_new_lambdas(new_lambdas)
    return new_lambdas


class MathSolver:

    def __init__(self, config_path="../problem_descriptors/slicing_params.json"):

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件路径错误: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.rb_max_dequeuing_rate = self.config["system_parameters"]["mu_max"]
        self.avg_airtime = self.config["system_parameters"]["airtime_needed_avg_us"]
        self.min_quantum = 1.0  # 定义一个极小的物理量子下限 (微秒)

        # 移除了self.model = GEKKO(...)
        # 因为在连续做 10 万道题的流水线中，GEKKO 模型必须在 solve() 内部进行初始化和清理，
        # 否则上一次解题的方程会残留到下一次，导致内存溢出和逻辑崩塌。

        # 原作者的这几个设置，将转移到内部求解函数中：
        # self.model.options.SOLVER = 1
        # self.model.options.IMODE = 3



        # Additional options
        # self.model.options.IMODE = 3

        # self.model.options.MAX_ITER = 5000000

        # Initialize the number of major iterations required to find a solution
        # self.model.solver_options = ['minlp_maximum_iterations 500', \
        #                             # minlp iterations with integer solution
        #                             'minlp_max_iter_with_int_sol 10', \
        #                             # treat minlp as nlp
        #                             'minlp_as_nlp 0', \
        #                             # nlp sub-problem max iterations
        #                             'nlp_maximum_iterations 500', \
        #                             # 1 = depth first, 2 = breadth first
        #                             'minlp_branch_method 2', \
        #                             # maximum deviation from whole number
        #                             'minlp_integer_tol 20', \
        #                             # covergence tolerance
        #                            'minlp_gap_tol 20']

    # 运行原作者的 while 删约束试错逻辑
    def solve(self, traffic_data):
        start_time = time.perf_counter()

        qos_constraints_to_remove = 0
        num_slices = len(traffic_data['lambdas'])
        success = False
        q_opt = None

        # 复原原作者 Pedro 的暴力试错逻辑
        while not success and qos_constraints_to_remove <= num_slices:
            success, q_opt = self._original_solve(
                traffic_data=traffic_data,
                qos_constraints_to_remove=qos_constraints_to_remove
            )

            if not success:
                qos_constraints_to_remove += 1

        cost_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            'success': success,
            'Q_opt': np.array(q_opt) if success else traffic_data['lambdas'] / np.sum(traffic_data['lambdas']),
            'constraints_removed': qos_constraints_to_remove if success else qos_constraints_to_remove - 1,
            'solve_time_ms': cost_time_ms
        }

    def _original_solve(self,
              num_experiment=0,
              num_slices=0,
              compute_new_lambdas_flag=False,
              lambda_distribution=None,
              lambda_gap=None,
              qos_constraints_to_remove=None,
              problem_handler=None,
              experiment_handler=None,
              objective_function=True,
              traffic_data=None):

        if traffic_data is not None:
            num_slices = len(traffic_data['lambdas'])

            class DummyProblemHandler: pass

            problem_handler = DummyProblemHandler()
            problem_handler.lambda_s = traffic_data['lambdas'].tolist()
            problem_handler.avg_airtimes = [self.avg_airtime] * num_slices
            problem_handler.rb_max_dequeuing_rate = self.rb_max_dequeuing_rate
            problem_handler.min_quantum = self.min_quantum
            problem_handler.qos_delay = (traffic_data['etas'] / 1000.0).tolist()
            psis = traffic_data['psis'].tolist()

        # Set this to false in case original problem lambdas wanted
        if compute_new_lambdas_flag and lambda_distribution is not None:
            if lambda_distribution == 'progressive' or lambda_distribution == 'other':
                problem_handler.lambda_s = compute_progressive_lambdas(num_slices=num_slices,
                                                                       rb_max_dequeuing_rate=problem_handler.rb_max_dequeuing_rate,
                                                                       lambda_factor=0.01,
                                                                       lambda_gap=lambda_gap)
            elif lambda_distribution == 'equal':
                problem_handler.lambda_s = compute_equal_lambdas(num_slices=num_slices,
                                                                 rb_max_dequeuing_rate=problem_handler.rb_max_dequeuing_rate,
                                                                 lambda_gap=lambda_gap)
            elif lambda_distribution == 'random':
                problem_handler.lambda_s = compute_random_lambdas(num_slices=num_slices,
                                                                  rb_max_dequeuing_rate=problem_handler.rb_max_dequeuing_rate,
                                                                  lambda_gap=lambda_gap)
        # 删除输出，防止冲爆控制台
        #     else:
        #         print('Lambda distribution not valid!')
        #
        # print(problem_handler)
        # print('Slices to be allocated:', num_slices)
        # print('SUM of the lambdas to be solved: ' + str(sum(problem_handler.lambda_s[0:num_slices])) + ' pps')

        # Initialize variables
        q_s = []  # Quantum values in slice s
        mu_s_pkt = []  # Packet dequeuing rate in slice s
        mu_s = []  # Dequeuing rate in slice s
        p_s = []  # Service utilization in slice s
        l_q_s = []  # Average number of bytes in slice s
        w_q_s = []  # Average waiting time in in slice s

        self.model = GEKKO(remote=False)
        self.model.options.SOLVER = 1  # APOPT 求解器

        for i in range(num_slices):
            # Quantums
            q_s.append(self.model.Var(lb=problem_handler.min_quantum,
                                      ub=problem_handler.avg_airtimes[i],
                                      # value=problem_handler.avg_airtimes[i] / 2,
                                      # Intial guess of 6000 us (12000 us / 2)
                                      # 改为 q_s_i 格式命名
                                      name='q_s_' + str(i)))

            # Packet dequeuing rate
            mu_s_pkt.append(self.model.Intermediate(q_s[i] / problem_handler.avg_airtimes[i],
                                                    name='mu_s_pkt_' + str(i)))

        for i in range(num_slices):
            # Dequeuing rate
            mu_s.append(self.model.Intermediate((mu_s_pkt[i] / sum(mu_s_pkt)) * problem_handler.rb_max_dequeuing_rate,
                                                name='mu_s_' + str(i)))

            # Service utilization
            p_s.append(self.model.Intermediate(problem_handler.lambda_s[i] / mu_s[i],
                                               name='p_s_' + str(i)))

            # Average service rate in the system
            l_q_s.append(self.model.Intermediate((p_s[i] * p_s[i]) / (1 - p_s[i]),
                                                 name='l_q_s_' + str(i)))

            # Average time of the system
            w_q_s.append(self.model.Intermediate(l_q_s[i] / problem_handler.lambda_s[i],
                                                 name='t_s_' + str(i)))

        # Equations (Constraints)
        for i in range(num_slices):
            self.model.Equation(mu_s[i] >= problem_handler.lambda_s[i])

        self.model.Equation(sum(mu_s) <= problem_handler.rb_max_dequeuing_rate)

        for i in range(num_slices):
            self.model.Equation(p_s[i] < 1)

            # Average time of the system have to be smaller than QoS delay
            if qos_constraints_to_remove is not None:
                if i >= qos_constraints_to_remove:
                    if problem_handler.qos_delay[i] is not None:
                        self.model.Equation(w_q_s[i] <= problem_handler.qos_delay[i])
        """
        传统基准算法的局限性与修正 (Limitations and Rectification of the Baseline Algorithm)在重构文献 [1] 提出的非线性运筹学求解器时，
        本研究发现其原始目标函数 min(sum(W_q)) 存在“无差别延迟最小化”的理论缺陷。
        在多 SLA 共存的网络切片场景中，未加权的目标函数会引导优化器在拥塞时牺牲 URLLC 等低延迟敏感业务，以换取 BE 等大吞吐量业务的延迟改善，
        这违背了 5G/6G 切片严格的优先级隔离原则。为了保证基准测试在现代通信场景下的公平性与科学性，本研究对基准求解器进行了最小化必要修正：
        引入业务紧急度权重 ψ (URLLC=10, eMBB=2, BE=1)，将目标函数重构为加权时延最小化 min(sum(ψ_i*W_{q,i}))。
        这一修正不仅保留了原作者非线性排队论模型的核心物理逻辑，同时迫使传统求解器具备了 SLA 感知的“优雅降级 (Graceful Degradation)”能力，
        为 AI 模型的评估提供了更具挑战性且符合物理真理的 Ground Truth。
        """
        # Objectives are always minimized (maximization is possible by multiplying the objective by -1)
        # if objective_function:
        #     obj = (sum(w_q_s))
        #     self.model.Obj(obj)
        # 修正了原作者 sum(w_q_s) 的无差别缺陷，引入 SLA 紧急度权重 psis
        if objective_function:
            # 原代码: obj = (sum(w_q_s))
            obj = self.model.sum([psis[i] * w_q_s[i] for i in range(num_slices)])
            self.model.Obj(obj)

        # self.model.open_folder()

        try:
            # self.model.solve(disp=True, debug=True)  # Solve
            # 关闭 disp，防止刷爆终端
            self.model.solve(disp=False, debug=False)

            # for i in range(num_slices):
            #     print('q_s      ' + str(i) + ': ' + str(q_s[i].value))
            #
            # for i in range(num_slices):
            #     print('mu_s_pkt ' + str(i) + ': ' + str(mu_s_pkt[i].value))
            #
            # for i in range(num_slices):
            #     print('mu_s     ' + str(i) + ': ' + str(mu_s[i].value))
            #
            # for i in range(num_slices):
            #     print('p_s      ' + str(i) + ': ' + str(p_s[i].value))
            #
            # for i in range(num_slices):
            #     print('l_q_s      ' + str(i) + ': ' + str(l_q_s[i].value))

            success = True
            q_opt = []

            # 删掉原作者的 print，直接获取答案
            for i in range(num_slices):
                q_opt.append(mu_s[i].value[0] / problem_handler.rb_max_dequeuing_rate)


            # if not objective_function:
            #     sum_of_ts = 0
            # for i in range(num_slices):
            #     print('w_q_s      ' + str(i) + ': ' + str(w_q_s[i].value))
            #     if not objective_function:
            #         sum_of_ts += w_q_s[i].value[0]
            #
            # if not objective_function:
            #     objective = sum_of_ts
            # else:
            #     objective = self.model.options.objfcnval

            # print('Objective: ' + str(objective))
            # print('Solve time: ' + str(self.model.options.SOLVETIME))

            # experiment_handler.write_results_into_file(num_experiment=num_experiment,
            #                                            num_slices=num_slices,
            #                                            objective=objective,
            #                                            lambdas_stdev=np.asarray(
            #                                                problem_handler.lambda_s[0:num_slices]).std(),
            #                                            solve_time=self.model.options.SOLVETIME,
            #                                            qos_constraints_removed=qos_constraints_to_remove)
            # return True
        except Exception:
            # print("An exception occurred during problem solving!")
            # return False


            # from gekko.apm import get_file
            # print(self.model._server)
            # print(self.model._model_name)
            # f = get_file(self.model._server, self.model._model_name, 'infeasibilities.txt')
            # f = f.decode().replace('\r', '')
            # with open('infeasibilities.txt', 'w') as fl:
            #     fl.write(str(f))

            # 如果系统极度拥塞导致 Gekko 无解报错，标记为失败
            success = False
            q_opt = None
        finally:
            # 清理缓存，防止撑爆硬盘
            self.model.cleanup()

        # 把 success 状态和答案 q_opt 一起返回给外层的 while 循环
        return success, q_opt



# 以下代码是原作者 Pedro 用来跑他自己毕业论文实验的代码，所以删掉
# def main():
#     """
#         Experiment Options
#     """
#     # Experiment flag
#     experiment = True
#
#     # Plot results flag
#     plot_results = False
#
#     '''
#         1- ADOPT: is generally the best when warm-starting from a prior solution or when the number of degrees of
#         freedom (Number of Variables - Number of Equations) is less than 2000
#
#         2- BPOPT has been found to be the best for systems biology applications.
#
#         3- IPOPT is generally the best for problems with large numbers of degrees of freedom or when starting without
#         a good initial guess.
#     '''
#     solver = 1
#     remote = True
#     objective_function_flag = True  # True or False in case feasibility want to be computed
#
#     if objective_function_flag:
#         target = 'optimality'
#     else:
#         target = 'feasibility'
#
#     if remote:
#         server = 'remote'
#     else:
#         server = 'local'
#
#     if solver == 1:
#         solver_name = "APOPT"
#     elif solver == 3:
#         solver_name = "IPOPT"
#     else:
#         solver_name = "ALL"
#
#     """
#     Lambda distributions:
#         progressive:    progressive lambda distribution
#         equal:          all lambdas are equal
#         other:          eliminate qos constraints one by one
#         custom:         keep the original problem descriptor
#         random:         generate random lambda distribution
#     """
#     lambda_distribution = 'equal'  # progressive, equal, random, other, or custom to keep the original problem
#     lambda_gap = 0.5
#
#     sleep = 1
#     from_slice = 1
#     until_slice = 7  # Set until desired slice number + 1
#     repeat = 1
#
#     # Load problem data and initialize results file
#     problem_handler = ProblemHandler(problem_filename="../../problem_descriptors/slicing_problem.json")
#
#     if experiment:
#         for num_slices in range(from_slice, until_slice):
#
#             qos_constraints_to_remove = 0  # Starts with 0, 1, 2,..
#
#             experiment_handler = ExperimentHandler(
#                 results_filename='results/' + str(server) + '/' + str(solver_name) + '/' + str(lambda_distribution) +
#                                  '/' + str(target) +
#                                  '/raw_results_SOL_' + str(solver_name) +
#                                  '_SLC_' + str(num_slices) +
#                                  '_SER_' + str(server) + '.csv')
#
#             # Init results file
#             experiment_handler.init_results_file()
#
#             for num_experiment in range(repeat):
#                 # Initialize problem
#                 problem = SlicingProblem(name="Optimal WiFi Slicing",
#                                          remote=remote,
#                                          solver=solver)
#
#                 # If we want to remove qos constraints one by one...
#                 if lambda_distribution == 'other':
#                     # Solve the problem with all QoS constraints until 0
#                     while not problem.solve(num_experiment=num_experiment,
#                                             num_slices=num_slices,
#                                             compute_new_lambdas_flag=True,
#                                             lambda_distribution=lambda_distribution,
#                                             lambda_gap=lambda_gap,
#                                             qos_constraints_to_remove=qos_constraints_to_remove,
#                                             problem_handler=problem_handler,
#                                             experiment_handler=experiment_handler,
#                                             objective_function=objective_function_flag):
#                         print('Solve failed for ' + str(num_slices) + ' with ' + str(qos_constraints_to_remove) + '!')
#
#                         qos_constraints_to_remove += 1
#
#                         # unfeasible...
#                         if qos_constraints_to_remove > num_slices:
#                             break
#
#                         # Reinitialise the problem
#                         problem = SlicingProblem(name="Optimal WiFi Slicing",
#                                                  remote=remote,
#                                                  solver=solver)
#
#                         print('Lets try with' + str(qos_constraints_to_remove) + ' constraints now!')
#
#                 else:
#                     # Just solve the problem
#                     if not problem.solve(num_experiment=num_experiment,
#                                          num_slices=num_slices,
#                                          compute_new_lambdas_flag=True,
#                                          lambda_distribution=lambda_distribution,
#                                          lambda_gap=lambda_gap,
#                                          problem_handler=problem_handler,
#                                          experiment_handler=experiment_handler,
#                                          objective_function=objective_function_flag):
#                         break
#                 time.sleep(sleep)
#
#             # Close results file
#             experiment_handler.close_results_file()
#     os.system('say Pedro, your wifi experiment has finished!')
#
#     if plot_results:
#         for num_slices in range(from_slice, until_slice):
#             draw_line_graph_with_multiple_y_axis(filename='results/' + str(server) + '/' + str(solver_name) + '/' +
#                                                           str(lambda_distribution) + '/' + str(target) +
#                                                           '/raw_results_SOL_' + str(solver_name) +
#                                                           '_SLC_' + str(num_slices) +
#                                                           '_SER_' + str(server) + '.csv',
#                                                  title='Optimal WiFi Slicing (Remote=' +
#                                                        str(remote) + ', Solver=' + str(solver_name) +
#                                                        ', Num Slices=' + str(num_slices) + ')',
#                                                  x_axis='Experiment (#)',
#                                                  y1_axis='Objective',
#                                                  y2_axis='Solution Time (sec)')
#
#
# if __name__ == "__main__": main()