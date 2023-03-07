import math

from optimizer.algorithms import Algorithm
from optimizer.random_solution import get_random_solution
from optimizer.solution import Solution


class Greedy(Algorithm):
    def __init__(self, hparams, problem_size, comm) -> None:
        super().__init__(hparams, problem_size, comm)

    def run(self, kmax, evaluation_session):
        Sbest = get_random_solution(self.problem_size, evaluation_session)
        Ebest = Sbest.cost()
        neighbors = Sbest.get_neighbors()
        k = 0
        newBetterS = True

        print('Cost= ', Ebest, end=' ')
        path = [(Sbest, Ebest)]
        Sbest.display()

        while k < kmax and len(neighbors) > 0 and newBetterS:
            S1 = neighbors.pop()
            E1 = S1.cost()
            for S2 in neighbors:
                E2 = S2.cost()
                if E2 > E1:
                    S1 = S2
                    E1 = E2
            if E1 > Ebest:
                Sbest = S1
                Ebest = E1
                neighbors = Sbest.get_neighbors()
                path.append((Sbest, Ebest))

                print('New best:', end=' ')
                Sbest.display()
                print('Actual Cost: ' + str(Ebest))

            else:
                newBetterS = False
                print("\nNo better element. End of the loop")

            k = k+1
        print("End of the loop via number of iterations")
        return Sbest, Ebest, path


class TabuGreedy(Algorithm):
    def __init__(self, hparams, problem_size, comm) -> None:
        super().__init__(hparams, problem_size, comm)
        self.register_hyperparameter('n_tabu', 5)
        self.parse_hyperparameters()

    def run(self, kmax, evaluation_session):
        N_Tabu = self.hparams['n_tabu']
        Sbest = get_random_solution(self.problem_size, evaluation_session)
        Ebest = Sbest.cost()
        neighbors = Sbest.get_neighbors()
        k = 0
        newBetterS = True

        visited = []

        print('Cost= ', Ebest, end=' ')
        path = [(Sbest, Ebest)]
        Sbest.display()

        while k < kmax and newBetterS:
            S1, E1 = TabuFindBest(neighbors, visited)
            if E1 > Ebest:
                Sbest = S1
                Ebest = E1
                visited = FifoAdd(Sbest, visited, N_Tabu)

                path.append((Sbest, Ebest))

                print('New best:', end=' ')
                Sbest.display()
                print('Actual Cost: ' + str(Ebest))

                neighbors = Sbest.get_neighbors()

            else:
                newBetterS = False
                print("\nNo better element. End of the loop")

            k = k+1

        print("End of the loop via number of iterations")
        return Sbest, Ebest, path


class parallelgreedy(Algorithm):
    def __init__(self, hparams, problem_size) -> None:
        super().__init__(hparams, problem_size)

    def run(self, kmax):

        Sbest = get_random_solution(self.problem_size)
        Ebest = Sbest.cost()
        neighbors = Sbest.get_neighbors()
        n = len(neighbors)
        k = 0
        newBetterS = True
        tabE = [0 for i in range(n)]
        print('Cost= ', Ebest, end=' ')
        path = [(Sbest, Ebest)]
        Sbest.display()

        while k < kmax and n > 0 and newBetterS:
            for i in range(n):
                tabE[i] = neighbors[i].cost()

            j = 0
            for i in range(n):
                if tabE[i] > tabE[j]:
                    j = i
            E1 = tabE[j]
            S1 = neighbors[j]

            if E1 > Ebest:
                Sbest = S1
                Ebest = E1
                neighbors = Sbest.get_neighbors()
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


def FifoAdd(Sbest, Ltabu, TabuSize=10):
    if len(Ltabu) == TabuSize:
        Ltabu.pop(0)
    Ltabu.append(Sbest)

    print("Tabu List: [", end=" ")
    for i in Ltabu:
        if i == Ltabu[-1]:
            print(Sbest.olevel, Sbest.simd, Sbest.problem_size_x, Sbest.problem_size_y, Sbest.problem_size_z,
                  Sbest.nthreads, Sbest.thrdblock_x, Sbest.thrdblock_y, Sbest.thrdblock_z, end=' ')
            print(']')
        else:
            print(Sbest.olevel, Sbest.simd, Sbest.problem_size_x, Sbest.problem_size_y, Sbest.problem_size_z,
                  Sbest.nthreads, Sbest.thrdblock_x, Sbest.thrdblock_y, Sbest.thrdblock_z, end=', ')

    return Ltabu


def TabuFindBest(Lneigh, Ltabu):
    E1 = -math.inf
    S1 = None
    for S2 in Lneigh:
        if S2 not in Ltabu:
            E2 = S2.cost()
            if E2 > E1:
                S1 = S2
                E1 = E2
    return S1, E1
