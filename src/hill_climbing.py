from solution import Solution
import random

from random_solution import get_random_solution
from algorithm import Algorithm

from mpi4py import MPI

import subprocess
import os 
import sys

class HillClimbing(Algorithm):
    def __init__(self, hparams, problem_size) -> None:
        super().__init__(hparams, problem_size)
        self.comm = MPI.COMM_WORLD        
        
    def run(self, num_steps):
        Sbest = get_random_solution(self.problem_size)
        Ebest = Sbest.cost()
        neighbors = Sbest.get_neighbors()
        k = 0
        print('Initial:', end=' ')
        Sbest.display()
        print('Cost= ', Ebest, end=' ')
        path = [(Sbest, Ebest)]
        Sbest.display()
        while k < num_steps and len(neighbors) > 0:
            selected_index = random.randint(0, len(neighbors)-1)
            S_new = neighbors[selected_index]
            neighbors.pop(selected_index)
            E_new = S_new.cost()
            print('Cost= ', E_new, end=' ')
            S_new.display()
            if E_new > Ebest:
                print('New best:', end=' ')
                Ebest = E_new
                Sbest = S_new
                Sbest.display()
                path.append((Sbest, Ebest))
                neighbors = Sbest.get_neighbors()
            k += 1
        return Sbest, Ebest, path