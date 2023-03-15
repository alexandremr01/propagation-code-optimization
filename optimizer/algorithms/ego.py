import math

from optimizer.algorithms import Algorithm
from optimizer.random_solution import get_random_solution
from optimizer.solution import Solution
from optimizer.solution_space import SolutionSpace

import numpy as np
from smt.applications import EGO
from smt.applications.ego import Evaluator
from smt.surrogate_models import KRG
import matplotlib.pyplot as plt
from optimizer.algorithms.curious_simulated_annealing import group_particles, ungroup_particles


def function_test_1d(x):
    # function xsinx
    import numpy as np

    x = np.reshape(x, (-1,))
    y = np.zeros(x.shape)
    y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
    return y.reshape((-1, 1))


class EfficientGlobalOptimization(Algorithm):

    def __init__(self, hparams, problem_size, comm, logger, optimize_problem_size) -> None:
        super().__init__(hparams, problem_size, comm, logger, optimize_problem_size)
        self.iteration = 0

    def run(self, n_iter, evaluation_session):
        self.evaluation_session = evaluation_session
        # slaves: go straight to slave execution, the only thing they will
        # do is wait for the scatter, then
        if self.comm.Get_rank() != 0:
            self.execute_slave()
            return None, None, None

        self.logger.write_info('Starting Efficient Global Optimization')
        n_parallel = 4
        n_start = 50
        xspecs = np.array(self.get_bounds())
        initial_solution = get_random_solution(self.problem_size)
        xdoe = [self.solution_to_x(initial_solution) for _ in range(4)]
        xdoe = np.atleast_2d(xdoe).T
        n_doe = xdoe.size

        criterion = "EI"  # 'EI' or 'SBO' or 'LCB'
        qEI = "KBUB"  # "KB", "KBLB", "KBUB", "KBRand"
        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            surrogate=KRG(print_global=False),
            xlimits=xspecs,
            n_parallel=n_parallel,
            qEI=qEI,
            n_start=n_start,
            evaluator=ParallelEvaluator(),
            random_state=42,
        )

        x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=lambda xs: self.parallel_cost_function(xs))

        self.comm.Barrier()  # alert all process to continue
        self.comm.scatter([None] * self.comm.Get_size(), root=0)  # signal to finish

        Sbest = self.x_to_solution(x_opt)
        Sbest.display()

        path = []
        return Sbest, Sbest.cost(evaluation_session), path


    def cost_function(self, x):
        solution = self.x_to_solution(x)
        cost = solution.cost(self.evaluation_session)
        print('Evaluating:', end=' ')
        solution.display()
        print('Cost: ', cost)
        return -cost

    def execute_slave(self):
        while True:
            self.comm.Barrier()
            # receive solutions to evaluate
            solutions = self.comm.scatter(None, root=0)
            if solutions is None:  # signal to stop
                return
            costs = self.list_costs(solutions)
            # send costs back
            self.comm.gather(costs, root=0)
            self.iteration = self.iteration + 1

    def list_costs(self, data):
        costs = []
        for x in data:
            solution = self.x_to_solution(x)
            cost = solution.cost(self.evaluation_session)
            self.logger.write_msg(
                self.iteration + 1, solution.cost(self.evaluation_session), solution.get_compilation_flags(),
                flair=None,
            )

            costs.append(-cost)
        return costs

    def parallel_cost_function(self, x):
        # make list of N_process lists of arrays
        grouped_x = group_particles(x, self.comm.Get_size())
        # alert all process to continue
        self.comm.Barrier()
        # send data to each process
        data = self.comm.scatter(grouped_x, root=0)
        # calculate own costs
        costs = self.list_costs(data)
        # aggregate costs from all processes
        costs = self.comm.gather(costs, root=0)
        self.iteration = self.iteration + 1
        y = ungroup_particles(costs)
        return np.atleast_2d(y).reshape((-1,1))

    def x_to_solution(self, x):
        x_parsed = [int(np.floor(xi)) for xi in x]
        return Solution(
            olevel=SolutionSpace.o_levels[x_parsed[0]],
            simd=SolutionSpace.simds[x_parsed[1]],
            problem_size_x=self.problem_size[0],
            problem_size_y=self.problem_size[1],
            problem_size_z=self.problem_size[2],
            nthreads=16,  # fixed
            thrdblock_x=self.problem_size[0],  # fixed
            thrdblock_y=SolutionSpace.threadblocks[x_parsed[2]],
            thrdblock_z=SolutionSpace.threadblocks[x_parsed[3]],
        )

    def solution_to_x(self, solution):
        return [
            SolutionSpace.o_levels.index(solution.olevel),
            SolutionSpace.simds.index(solution.simd),
            SolutionSpace.threadblocks.index(solution.thrdblock_y),
            SolutionSpace.threadblocks.index(solution.thrdblock_z),
        ]

    def get_bounds(self):
        return [[0.5, len(SolutionSpace.o_levels) - 0.5],
                [0.5, len(SolutionSpace.simds) - 0.5],
                [0.5, len(SolutionSpace.threadblocks) - 0.5],
                [0.5, len(SolutionSpace.threadblocks) - 0.5]]

class ParallelEvaluator(Evaluator):
    def run(self, fun, x):
        return fun(x)
