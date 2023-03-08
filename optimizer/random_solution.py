import numpy as np

from optimizer.solution import Solution
from optimizer.solution_space import SolutionSpace

def get_random_solution(problem_size):
    return Solution(
        olevel=np.random.choice(SolutionSpace.o_levels),
        simd=np.random.choice(SolutionSpace.simds),
        problem_size_x=problem_size[0],
        problem_size_y=problem_size[1],
        problem_size_z=problem_size[2],
        nthreads=np.random.choice(SolutionSpace.nthreads),
        # fixed at problem size
        thrdblock_x=np.random.choice(SolutionSpace.threadblocksx),
        thrdblock_y=np.random.choice(SolutionSpace.threadblocks),
        thrdblock_z=np.random.choice(SolutionSpace.threadblocks),
    )

