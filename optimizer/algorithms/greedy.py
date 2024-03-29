import math

from optimizer.algorithms import Algorithm
from optimizer.random_solution import get_random_solution
from optimizer.solution import Solution


class Greedy(Algorithm):
    def __init__(self, hparams, problem_size, comm, logger, optimize_problem_size) -> None:
        super().__init__(hparams, problem_size, comm, logger, optimize_problem_size)

    def run(self, kmax, evaluator):
        self.logger.write_info('Starting greedy hill climbing')
        Sbest = get_random_solution(self.problem_size)
        Ebest = evaluator.cost(Sbest)
        neighbors = Sbest.get_neighbors(self.optimize_problem_size)
        k = 0
        newBetterS = True
        path = [(Sbest, Ebest)]
        self.logger.write_msg(
            k, evaluator.get_counter(), Ebest, Sbest.get_compilation_flags(), flair='Initial'
        )

        while k < kmax and len(neighbors) > 0 and newBetterS:
            S1 = neighbors.pop()
            E1 = evaluator.cost(S1)
            for S2 in neighbors:
                E2 = evaluator.cost(S2)
                if E2 > E1:
                    S1 = S2
                    E1 = E2
            if E1 > Ebest:
                Sbest = S1
                Ebest = E1
                neighbors = Sbest.get_neighbors(self.optimize_problem_size)
                path.append((Sbest, Ebest))
                self.logger.write_msg(
                    k+1, evaluator.get_counter(), Ebest, Sbest.get_compilation_flags(),
                )
            else:
                newBetterS = False
                self.logger.write_info("No better element. End of the loop")

            k = k+1
        self.logger.write_info("End of the loop via number of iterations")
        return Sbest, Ebest, path    

class TabuGreedy(Algorithm):
    def __init__(self, hparams, problem_size, comm, logger, optimize_problem_size) -> None:
        super().__init__(hparams, problem_size, comm, logger, optimize_problem_size)
        self.register_hyperparameter('n_tabu', 5)
        self.parse_hyperparameters()

    def run(self, kmax, evaluator):
        self.logger.write_info('Starting tabu_greedy hill climbing')
        N_Tabu = self.hparams['n_tabu']
        Sbest = get_random_solution(self.problem_size)
        Ebest = evaluator.cost(Sbest)
        neighbors = Sbest.get_neighbors(self.optimize_problem_size)
        k = 0
        path = [(Sbest, Ebest)]
        self.logger.write_msg(
            k, evaluator.get_counter(), Ebest, Sbest.get_compilation_flags(), flair='Initial'
        )

        newBetterS = True
        visited = []

        while k < kmax and newBetterS:
            S1, E1 = TabuFindBest(neighbors, visited, evaluator)
            if E1 > Ebest:
                Sbest = S1
                Ebest = E1
                visited = FifoAdd(self.logger, Sbest, visited, N_Tabu)

                path.append((Sbest, Ebest))

                self.logger.write_msg(
                    k+1, evaluator.get_counter(), Ebest, Sbest.get_compilation_flags(),
                )

                neighbors = Sbest.get_neighbors(self.optimize_problem_size)

            else:
                newBetterS = False
                self.logger.write_info("No better element. End of the loop")

            k = k+1

        print("End of the loop via number of iterations")
        return Sbest, Ebest, path


class parallelgreedy(Algorithm):
    def __init__(self, hparams, problem_size) -> None:
        super().__init__(hparams, problem_size)

    def run(self, kmax, evaluator):

        Sbest = get_random_solution(self.problem_size)
        Ebest = evaluator.cost(Sbest)
        neighbors = Sbest.get_neighbors(self.optimize_problem_size)
        n = len(neighbors)
        k = 0
        newBetterS = True
        tabE = [0 for i in range(n)]
        print('Cost= ', Ebest, end=' ')
        path = [(Sbest, Ebest)]
        Sbest.display()

        while k < kmax and n > 0 and newBetterS:
            for i in range(n):
                tabE[i] = evaluator.cost(neighbors[i])

            j = 0
            for i in range(n):
                if tabE[i] > tabE[j]:
                    j = i
            E1 = tabE[j]
            S1 = neighbors[j]

            if E1 > Ebest:
                Sbest = S1
                Ebest = E1
                neighbors = Sbest.get_neighbors(self.optimize_problem_size)
                path.append((Sbest, Ebest))
                print('New best:', end=' ')
                Sbest.display()
                print('Actual Cost: ' + str(Ebest))

            else:
                newBetterS = False
                print("\nNo better element. End of the loop")

            k = k+1
        print("End of the loop via number of iterations")
        return Sbest, path


def FifoAdd(logger, Sbest, Ltabu, TabuSize=10):
    if len(Ltabu) == TabuSize:
        Ltabu.pop(0)
    Ltabu.append(Sbest)

    logger.write_info("Tabu List:")
    for i in Ltabu:
            logger.write_raw("\t" + i.get_compilation_flags())

    return Ltabu


def TabuFindBest(Lneigh, Ltabu, evaluator):
    E1 = -math.inf
    S1 = None
    for S2 in Lneigh:
        if S2 not in Ltabu:
            E2 = evaluator.cost(S2)
            if E2 > E1:
                S1 = S2
                E1 = E2
    return S1, E1
