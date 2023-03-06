import os
import sys
import subprocess
import random

from mpi4py import MPI

from optimizer.solution import Solution
from optimizer.random_solution import get_random_solution
from optimizer.algorithms import Algorithm

class HillClimbing(Algorithm):
    def __init__(self, hparams, problem_size, comm, logger) -> None:
        super().__init__(hparams, problem_size, comm, logger)
        
    def run(self, num_steps, evaluation_session):
        self.logger.write_info('Starting hill_climbing')
        Sbest = get_random_solution(self.problem_size, evaluation_session)
        Ebest = Sbest.cost()
        neighbors = Sbest.get_neighbors()
        k = 0
        path = [(Sbest, Ebest)]
        self.logger.write_msg(
            k, Ebest, Sbest.get_compilation_flags(),
        )
        while k < num_steps and len(neighbors) > 0:
            selected_index = random.randint(0, len(neighbors)-1)
            S_new = neighbors[selected_index]
            neighbors.pop(selected_index)
            E_new = S_new.cost()
            if E_new > Ebest:
                log_flair = 'New best!'
                Ebest = E_new
                Sbest = S_new
                path.append((Sbest, Ebest))
                neighbors = Sbest.get_neighbors()
            else:
                log_flair = None
            k += 1

            self.logger.write_msg(
                k, E_new, S_new.get_compilation_flags(), flair=log_flair
            )
        return Sbest, Ebest, path
