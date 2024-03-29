import random
import math

from optimizer.solution import Solution
from optimizer.random_solution import get_random_solution
from optimizer.algorithms import Algorithm

class LocalConditionnalAcceptance(Algorithm):
    def __init__(self, hparams, problem_size, comm, logger, optimize_problem_size) -> None:
        super().__init__(hparams, problem_size, comm, logger, optimize_problem_size)
        self.register_hyperparameter('t0', 100)
        self.register_hyperparameter('lambda', 0.9)
        self.parse_hyperparameters()

        self.T0 = self.hparams['t0']
        # TODO: current temperature function is hard coded
        self.f = lambda x: self.hparams['lambda']*x

    def run(self, kmax, evaluator):
        self.logger.write_info('Starting simulated_annealing')
        T0 = self.T0
        f = self.f
        S_best = get_random_solution(self.problem_size)
        E_best = evaluator.cost(S_best)
        S = S_best
        E = E_best
        neighbors = S_best.get_neighbors(self.optimize_problem_size)
        path = [(S_best, E_best)]
        T = T0
        k = 0
        self.logger.write_msg(
            k, evaluator.get_counter(), E, S.get_compilation_flags(), flair='Initial'
        )
        while k < kmax > 0:
            selected_index = random.randint(0, len(neighbors)-1)
            S_new = neighbors[selected_index]
            E_new = evaluator.cost(S_new)
            if E_new > E or random.uniform(0, 1) < math.exp((E_new-E)/T):
                if E_new <= E:
                    log_flair = 'Risky choice !'
                S = S_new
                E = E_new
                neighbors = S.get_neighbors(self.optimize_problem_size)
                if E > E_best:
                    S_best = S
                    E_best = E
                    log_flair = 'New best'
                    path.append((S_best, E_best))
            else:
                log_flair = None
            T = f(T)
            k += 1
            self.logger.write_msg(
                k, evaluator.get_counter(), E_new, S_new.get_compilation_flags(), log_flair,
            )
        return S_best, E_best, path
