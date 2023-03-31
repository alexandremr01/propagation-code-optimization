import cma
import numpy as np

from optimizer.solution import Solution
from optimizer.evaluators import Simulator
from optimizer.random_solution import get_random_solution
from optimizer.solution_space import SolutionSpace
from optimizer.algorithms import Algorithm
from optimizer.algorithms.curious_simulated_annealing import group_particles, ungroup_particles


class CMAESAlgorithm(Algorithm):
    def __init__(self, hparams, problem_size, comm, logger, optimize_problem_size) -> None:
        if optimize_problem_size:
            raise Exception('CMAES not compatible with optimize problem size')
        super().__init__(hparams, problem_size, comm, logger, optimize_problem_size)
        self.problem_size_product = problem_size[0] * problem_size[1] * problem_size[2]
        self.iteration = 0

    def run(self, kmax, evaluator):
        self.evaluator = evaluator
        # slaves: go straight to slave execution, the only thing they will
        # do is wait for the scatter, then 
        if self.comm.Get_rank() != 0:
            self.execute_slave()
            return None, None, None

        initial_solution = get_random_solution(self.problem_size)
        x0 = self.solution_to_x(initial_solution)
        sigma0 = 1  # initial standard deviation to sample new solutions
        num_optimizing_variables = 6 if self.optimize_problem_size else 4
        options = {
            'bounds': self.get_bounds(),
            'integer_variables': list(range(num_optimizing_variables)),
            'maxfevals': kmax,
            'verbose': -9,
        }
        x, es = cma.fmin2(
            None,  # lambda x : self.cost_function(x),
            x0,
            sigma0,
            options,
            parallel_objective=lambda xs: self.parallel_cost_function(xs),
        )
        self.comm.Barrier()  # alert all process to continue
        self.comm.scatter([None] * self.comm.Get_size(), root=0)  # signal to finish
        print('Returned: ', x)
        Sbest = self.x_to_solution(x)
        Sbest.display()
        # TODO: how to manage path for CMAES?
        path = []
        return Sbest, evaluator.cost(Sbest), path

    def cost_function(self, x):  # currntly unuseed function
        solution = self.x_to_solution(x)
        cost = self.evaluator.cost(solution)
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
            cost = self.evaluator.cost(solution)
            self.logger.write_msg(
                self.iteration + 1, self.evaluator.get_counter(), cost, solution.get_compilation_flags(),
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
        return ungroup_particles(costs)

    def x_to_solution(self, x):
        x_parsed = [int(np.floor(xi)) for xi in x]
        problem_size_x, problem_size_y, problem_size_z = self.problem_size
        if self.optimize_problem_size:
            problem_size_x = SolutionSpace.problem_size[x_parsed[4]]
            problem_size_y = SolutionSpace.problem_size[x_parsed[5]]
            problem_size_z = self.problem_size_product // (problem_size_x * problem_size_y)
        return Solution(
            olevel=SolutionSpace.o_levels[x_parsed[0]],
            simd=SolutionSpace.simds[x_parsed[1]],
            nthreads=16,  # fixed
            thrdblock_x=problem_size_x,  # fixed
            thrdblock_y=SolutionSpace.threadblocks[x_parsed[2]],
            thrdblock_z=SolutionSpace.threadblocks[x_parsed[3]],
            problem_size_x=problem_size_x,
            problem_size_y=problem_size_y,
            problem_size_z=problem_size_z,
        )

    def solution_to_x(self, solution):
        x = [
            SolutionSpace.o_levels.index(solution.olevel),
            SolutionSpace.simds.index(solution.simd),
            SolutionSpace.threadblocks.index(solution.thrdblock_y),
            SolutionSpace.threadblocks.index(solution.thrdblock_z),
        ]
        if self.optimize_problem_size:
            x += [
                SolutionSpace.problem_size.index(solution.problem_size_x),
                SolutionSpace.problem_size.index(solution.problem_size_y),
            ]
        return x

    def get_bounds(self):
        upper_bounds = [
            len(SolutionSpace.o_levels) - 1e-3,
            len(SolutionSpace.simds) - 1e-3,
            len(SolutionSpace.threadblocks) - 1e-3,
            len(SolutionSpace.threadblocks) - 1e-3,
        ]
        if self.optimize_problem_size:
            upper_bounds += [
                len(SolutionSpace.problem_size) - 1e-3,
                len(SolutionSpace.problem_size) - 1e-3
            ]
        return [0, upper_bounds]
