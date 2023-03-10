import cma
import numpy as np

from optimizer.solution import Solution
from optimizer.evaluator import Simulator
from optimizer.random_solution import get_random_solution
from optimizer.solution_space import SolutionSpace
from optimizer.algorithms import Algorithm
from optimizer.algorithms.curious_simulated_annealing import group_particles, ungroup_particles

class CMAESAlgorithm(Algorithm):
    def __init__(self, hparams, problem_size, comm, logger) -> None:
        super().__init__(hparams, problem_size, comm, logger)
        self.iteration = 0

    def run(self, kmax, evaluation_session):
        self.evaluation_session = evaluation_session
        # slaves: go straight to slave execution, the only thing they will
        # do is wait for the scatter, then 
        if self.comm.Get_rank() != 0:
            self.execute_slave()
            return None, None, None

        initial_solution = get_random_solution(self.problem_size)
        x0 = self.solution_to_x(initial_solution)
        sigma0 = 1   # initial standard deviation to sample new solutions
        options = {
            'bounds': self.get_bounds(),
            'integer_variables': [0, 1, 2, 3],
            'maxfevals': kmax,
            'verbose': -9,
        }
        # TODO: pass parallel_objective instead of cost_function
        x, es = cma.fmin2(
            None, #lambda x : self.cost_function(x), 
            x0, 
            sigma0, 
            options,
            parallel_objective=lambda xs:self.parallel_cost_function(xs),
        )
        self.comm.Barrier() # alert all process to continue
        self.comm.scatter([None]*self.comm.Get_size(),root=0) # signal to finish
        print('Returned: ', x)
        Sbest = self.x_to_solution(x)
        Sbest.display()
        # TODO: how to manage path for CMAES?
        path = [ ]
        return Sbest, Sbest.cost(evaluation_session), path

    def cost_function(self, x): #currntly unuseed function
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
            solutions = self.comm.scatter(None,root=0)
            if solutions is None: # signal to stop
                return
            costs = self.list_costs(solutions)
            # send costs back
            self.comm.gather(costs,root=0)
            self.iteration = self.iteration + 1

    def list_costs(self, data):
        costs = [ ]
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
        data = self.comm.scatter(grouped_x,root=0) 
        # calculate own costs
        costs = self.list_costs(data) 
        # aggregate costs from all processes
        costs = self.comm.gather(costs,root=0)
        self.iteration = self.iteration + 1
        return ungroup_particles(costs)

    def x_to_solution(self, x):
        x_parsed = [int(np.floor(xi)) for xi in x]
        return Solution(
            olevel=SolutionSpace.o_levels[x_parsed[0]],
            simd=SolutionSpace.simds[x_parsed[1]],
            problem_size_x=self.problem_size[0],
            problem_size_y=self.problem_size[1],
            problem_size_z=self.problem_size[2],
            nthreads=16, # fixed 
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
        return [0, [
            len(SolutionSpace.o_levels)-1e-3, 
            len(SolutionSpace.simds)-1e-3, 
            len(SolutionSpace.threadblocks)-1e-3, 
            len(SolutionSpace.threadblocks)-1e-3, 
        ]]