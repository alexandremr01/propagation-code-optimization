import subprocess
import re
import threading
import os


class Simulator:
    def __init__(self, logger, run_counter=0, solutions_counter=0) -> None:
        logger.write_info('Simulator initialized')
        self.run_counter = run_counter  # counts how many solutions were evaluated in a run
        self.sol_counter = solutions_counter  # counts how many solutions were instanced in a run

    def sol_increase(self):
        self.sol_counter = self.sol_counter + 1

    def run_increase(self, num_evaluations):
        self.run_counter = self.run_counter + num_evaluations

    def display(self):
        print(str(self.sol_counter) + ' Solutions have been instanced. The function has been executed ' +
              str(self.run_counter) + ' times.')


class BaseEvaluator:
    def __init__(self, program_path, evaluation_session):
        self.program_path = program_path
        self.evaluation_session = evaluation_session

    def get_counter(self):
        return self.evaluation_session.run_counter


class NaiveEvaluator(BaseEvaluator):
    def __init__(self, program_path, evaluation_session):
        super().__init__(program_path, evaluation_session)

    def cost(self, solution, verbose=False, delete_file=True, num_evaluations=1, ignore_cache=False, affinity='balanced'):
        program_path = self.program_path
        if not ignore_cache and solution.calculated_cost is not None:
            return solution.calculated_cost
        self.evaluation_session.run_increase(num_evaluations)  # Increases in num_evaluations the counter of runs

        file_name = str(threading.get_ident())
        file_name_with_ext = f'{file_name}.exe'
        executable_path = f'{program_path}/bin/{file_name_with_ext}'

        result = subprocess.run(
            ['make', '-C', program_path, f'Olevel={solution.olevel}', f'simd={solution.simd}', 'last'],
            stdout=subprocess.DEVNULL,
            env=dict(os.environ, CONFIG_EXE_NAME=file_name_with_ext))
        if result.returncode != 0:
            raise Exception(f'Failed compiling: {result.returncode}')

        mean_throughput = 0
        new_environment = dict(os.environ, KMP_AFFINITY=affinity)
        for _ in range(num_evaluations):
            result = subprocess.run([executable_path,
                                     str(solution.problem_size_x),
                                     str(solution.problem_size_y),
                                     str(solution.problem_size_z),
                                     str(solution.nthreads), '100',
                                     str(solution.thrdblock_x),
                                     str(solution.thrdblock_y),
                                     str(solution.thrdblock_z)],
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

        solution.calculated_cost = mean_throughput
        return mean_throughput


class EnergyEvaluator:
    def __init__(self, program_path, evaluation_session):
        super().__init__(program_path, evaluation_session)

    # TODO: implement
    def cost(self, solution, verbose=False, delete_file=True, num_evaluations=1, ignore_cache=False):
        raise NotImplementedError
