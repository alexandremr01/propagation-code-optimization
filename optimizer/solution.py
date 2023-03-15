import subprocess
import re
import threading
import os
import numpy as np

from optimizer.solution_space import SolutionSpace


class Solution:
    def __init__(self, olevel, simd, problem_size_x, problem_size_y, problem_size_z, nthreads, thrdblock_x, thrdblock_y,
                 thrdblock_z) -> None:
        self.olevel = olevel
        self.simd = simd
        self.problem_size_x = problem_size_x
        self.problem_size_y = problem_size_y
        self.problem_size_z = problem_size_z
        self.nthreads = nthreads
        self.thrdblock_x = thrdblock_x
        self.thrdblock_y = thrdblock_y
        self.thrdblock_z = thrdblock_z
        self.calculated_cost = None

    def get_modified_copy(self, olevel=None, simd=None, problem_size_x=None, problem_size_y=None, problem_size_z=None,
                          nthreads=None, thrdblock_x=None, thrdblock_y=None, thrdblock_z=None):
        new_sol = Solution(self.olevel, self.simd, self.problem_size_x, self.problem_size_y, self.problem_size_z,
                           self.nthreads, self.thrdblock_x, self.thrdblock_y, self.thrdblock_z)
        if olevel is not None:
            new_sol.olevel = olevel
        if simd is not None:
            new_sol.simd = simd
        if problem_size_x is not None:
            new_sol.problem_size_x = problem_size_x
        if problem_size_y is not None:
            new_sol.problem_size_y = problem_size_y
        if problem_size_z is not None:
            new_sol.problem_size_z = problem_size_z
        if nthreads is not None:
            new_sol.nthreads = nthreads
        if thrdblock_x is not None:
            new_sol.thrdblock_x = thrdblock_x
        if thrdblock_y is not None:
            new_sol.thrdblock_y = thrdblock_y
        if thrdblock_z is not None:
            new_sol.thrdblock_z = thrdblock_z
        return new_sol

    def cost(self, evaluation_session, verbose=False, delete_file=True, num_evaluations=1):
        if self.calculated_cost is not None:
            return self.calculated_cost

        evaluation_session.run_increase(num_evaluations)  # Increases in num_evaluations the counter of runs

        file_name = str(threading.get_ident())
        file_name_with_ext = f'{file_name}.exe'
        executable_path = f'iso3dfd-st7/bin/{file_name_with_ext}'

        result = subprocess.run(['make', '-C', 'iso3dfd-st7', f'Olevel={self.olevel}', f'simd={self.simd}', 'last'],
                                stdout=subprocess.DEVNULL,
                                env=dict(os.environ, CONFIG_EXE_NAME=file_name_with_ext))
        if result.returncode != 0:
            raise Exception(f'Failed compiling: {result.returncode}')

        mean_throughput = 0
        new_environment = dict(os.environ, KMP_AFFINITY='scatter')
        for _ in range(num_evaluations):
            result = subprocess.run([executable_path,
                                     str(self.problem_size_x),
                                     str(self.problem_size_y),
                                     str(self.problem_size_z),
                                     str(self.nthreads), '100',
                                     str(self.thrdblock_x),
                                     str(self.thrdblock_y),
                                     str(self.thrdblock_z)],
                                    capture_output=True,
                                    env=new_environment)
            if result.returncode != 0:
                raise Exception(f'Failed executing: {result.returncode}')

            output = result.stdout
            m = re.search('throughput:\s+([\d\.]+)', str(output))
            throughput = m.group(1)
            try:
                mean_throughput += float(throughput)
            except:
                raise ValueError('throughput not a float')
            if verbose:
                print(output)

        if delete_file:
            result = subprocess.run(['rm', executable_path])
            if result.returncode != 0:
                raise Exception(f'Failed deleting: {result.returncode}')

        mean_throughput = round(mean_throughput / num_evaluations, 2)

        self.calculated_cost = mean_throughput
        return mean_throughput

    def get_neighbors(self, optimize_problem_size=True):
        neigh = []

        olevels = set(SolutionSpace.o_levels)
        olevels.remove(self.olevel)
        for level in olevels:
            neigh.append(self.get_modified_copy(olevel=level))

        simds = set(SolutionSpace.simds)
        simds.remove(self.simd)
        for simd in simds:
            neigh.append(self.get_modified_copy(simd=simd))

        n_thread_ix = SolutionSpace.nthreads.index(self.nthreads)
        if n_thread_ix > 0:
            new_n_thread = SolutionSpace.nthreads[n_thread_ix - 1]
            neighbor = self.get_modified_copy(nthreads=new_n_thread)
            neigh.append(neighbor)
        if n_thread_ix < len(SolutionSpace.nthreads) - 1:
            new_n_thread = SolutionSpace.nthreads[n_thread_ix + 1]
            neighbor = self.get_modified_copy(nthreads=new_n_thread)
            neigh.append(neighbor)

        thrdblock_y_ix = SolutionSpace.threadblocks.index(self.thrdblock_y)
        if thrdblock_y_ix > 0:
            new_thrdblock_y = SolutionSpace.threadblocks[thrdblock_y_ix - 1]
            neighbor = self.get_modified_copy(thrdblock_y=new_thrdblock_y)
            neigh.append(neighbor)
        if thrdblock_y_ix < len(SolutionSpace.threadblocks) - 1:
            new_thrdblock_y = SolutionSpace.threadblocks[thrdblock_y_ix + 1]
            neighbor = self.get_modified_copy(thrdblock_y=new_thrdblock_y)
            neigh.append(neighbor)

        thrdblock_z_ix = SolutionSpace.threadblocks.index(self.thrdblock_z)
        if thrdblock_z_ix > 0:
            new_thrdblock_z = SolutionSpace.threadblocks[thrdblock_z_ix - 1]
            neighbor = self.get_modified_copy(thrdblock_z=new_thrdblock_z)
            neigh.append(neighbor)
        if thrdblock_z_ix < len(SolutionSpace.threadblocks) - 1:
            new_thrdblock_z = SolutionSpace.threadblocks[thrdblock_z_ix + 1]
            neighbor = self.get_modified_copy(thrdblock_z=new_thrdblock_z)
            neigh.append(neighbor)

        if optimize_problem_size:
            problem_size_x_ix = SolutionSpace.problem_size.index(self.problem_size_x)
            problem_size_y_ix = SolutionSpace.problem_size.index(self.problem_size_y)
            problem_size_z_ix = SolutionSpace.problem_size.index(self.problem_size_z)
            if self._is_new_shape_available(problem_size_x_ix, problem_size_y_ix):
                neigh.append(self.get_modified_copy(
                    problem_size_x=SolutionSpace.problem_size[problem_size_x_ix - 1],
                    problem_size_y=SolutionSpace.problem_size[problem_size_y_ix + 1],
                ))
            if self._is_new_shape_available(problem_size_y_ix, problem_size_x_ix):
                neigh.append(self.get_modified_copy(
                    problem_size_x=SolutionSpace.problem_size[problem_size_x_ix + 1],
                    problem_size_y=SolutionSpace.problem_size[problem_size_y_ix - 1],
                ))
            if self._is_new_shape_available(problem_size_x_ix, problem_size_z_ix):
                neigh.append(self.get_modified_copy(
                    problem_size_x=SolutionSpace.problem_size[problem_size_x_ix - 1],
                    problem_size_z=SolutionSpace.problem_size[problem_size_z_ix + 1],
                ))
            if self._is_new_shape_available(problem_size_z_ix, problem_size_x_ix):
                neigh.append(self.get_modified_copy(
                    problem_size_x=SolutionSpace.problem_size[problem_size_x_ix + 1],
                    problem_size_z=SolutionSpace.problem_size[problem_size_z_ix - 1],
                ))
            if self._is_new_shape_available(problem_size_y_ix, problem_size_z_ix):
                neigh.append(self.get_modified_copy(
                    problem_size_y=SolutionSpace.problem_size[problem_size_y_ix - 1],
                    problem_size_z=SolutionSpace.problem_size[problem_size_z_ix + 1],
                ))
            if self._is_new_shape_available(problem_size_z_ix, problem_size_y_ix):
                neigh.append(self.get_modified_copy(
                    problem_size_y=SolutionSpace.problem_size[problem_size_y_ix + 1],
                    problem_size_z=SolutionSpace.problem_size[problem_size_z_ix - 1],
                ))

        return neigh

    def _is_new_shape_available(self, dim1_ix, dim2_ix):
        """return if it is possible to reduce dim1 and increase dim2"""
        return dim1_ix > 0 and dim2_ix < len(SolutionSpace.problem_size) - 1

    def get_random_neighbor(self):
        neighbors = self.get_neighbors()
        return np.random.choice(neighbors)

    def display(self):
        print(self.olevel, self.simd, self.problem_size_x, self.problem_size_y,
              self.problem_size_z, self.nthreads, self.thrdblock_x, self.thrdblock_y, self.thrdblock_z)

    def get_compilation_flags(self):
        return " ".join((self.olevel, self.simd, str(self.problem_size_x), str(self.problem_size_y),
                         str(self.problem_size_z), str(self.nthreads), str(self.thrdblock_x), str(self.thrdblock_y),
                         str(self.thrdblock_z)))
