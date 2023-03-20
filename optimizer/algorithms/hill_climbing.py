import os
import sys
import subprocess
import random

from mpi4py import MPI

from optimizer.solution import Solution
from optimizer.random_solution import get_random_solution
from optimizer.algorithms import Algorithm


class HillClimbing(Algorithm):
    def __init__(self, hparams, problem_size, comm, logger, optimize_problem_size) -> None:
        super().__init__(hparams, problem_size, comm, logger, optimize_problem_size)
        
    def run(self, num_steps, evaluator):
        self.logger.write_info('Starting hill_climbing')
        Sbest = get_random_solution(self.problem_size)
        Ebest = evaluator.cost(Sbest)
        neighbors = Sbest.get_neighbors(self.optimize_problem_size)
        k = 0
        path = [(Sbest, Ebest)]
        self.logger.write_msg(
            k, evaluator.get_counter(), Ebest, Sbest.get_compilation_flags(), flair='Initial'
        )
        while k < num_steps and len(neighbors) > 0:
            selected_index = random.randint(0, len(neighbors)-1)
            S_new = neighbors[selected_index]
            neighbors.pop(selected_index)
            E_new = evaluator.cost(S_new)
            if E_new > Ebest:
                log_flair = 'New best!'
                Ebest = E_new
                Sbest = S_new
                path.append((Sbest, Ebest))
                neighbors = Sbest.get_neighbors(self.optimize_problem_size)
            else:
                log_flair = None
            k += 1

            self.logger.write_msg(
                k, evaluator.get_counter(), E_new, S_new.get_compilation_flags(), flair=log_flair
            )
        if len(neighbors) <= 0:
            self.logger.write_info(
                'Algorithm exited: Best solution neighborhood was fully explored ')

        return Sbest, Ebest, path
