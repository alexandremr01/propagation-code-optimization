from solution import Solution
import random
import math
from algorithm import Algorithm

from random_solution import get_random_solution

class LocalConditionnalAcceptance(Algorithm):
    def __init__(self, hparams, problem_size) -> None:
        super().__init__(hparams, problem_size)
        self.T0 = hparams['t0']
        # TODO: current temperature function is hard coded
        self.f = lambda x : 0.9*x

    def run(self, kmax):
        T0 = self.T0
        f = self.f
        S_best = get_random_solution(self.problem_size)
        E_best = S_best.cost()
        S = S_best
        E = E_best
        neighbors = S_best.get_neighbors()
        T = T0
        k = 0
        while k < kmax > 0:
            selected_index = random.randint(0, len(neighbors)-1)
            S_new = neighbors[selected_index]
            E_new = S_new.cost()
            print('Cost= ', E_new, end=' ')
            S_new.display()
            path = [(S_best, E_best)]
            if E_new > E or random.uniform(0, 1) < math.exp((E_new-E)/T):
                if E_new <= E:
                    print('Risky choice !', end=' ')
                    S.display()
                S = S_new
                E = E_new
                neighbors = S.get_neighbors()
                if E > E_best:
                    S_best = S
                    E_best = E
                    print('New best:', end=' ')
                    S_new.display()
                    path.append((S_best, E_best))
            T = f(T)
            k += 1
        return S_best, E_best, path
