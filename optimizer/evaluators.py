import subprocess
import re
import threading
import os
import pandas as pd
import numpy as np

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

#To use this class you must be root
class EnergyEvaluator(BaseEvaluator): 
    def __init__(self, olevel, simd, problem_size_x, problem_size_y, problem_size_z, nthreads, thrdblock_x, thrdblock_y, thrdblock_z) -> None:
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
        self.calculated_energy = None

    def csv_to_energy(self,csv_path='temporary_csv_file.csv'):
        """takes a csv file from cpu_monitor and return the energy consumed"""
        idx = pd.Index([],dtype='int64')
        df = pd.read_csv(csv_path,sep=';')
        df = df.iloc[1: , :]

        sub_df = df.filter(regex=("^PW_*"))
        (max_row, max_col) = sub_df.shape
        for i in range(0, max_col):
            idx = idx.union(sub_df[sub_df.iloc[:,i]>8000.0].index)
        df.drop(idx, inplace=True)
        print('filter out {} rows'.format(idx.size))
            
        (max_row, max_col) = df.shape
        print('row: {}, col: {}'.format(max_row,max_col))
        df[df.select_dtypes(include=[np.number]).ge(0).all(1)]
        print('row: {}, col: {}'.format(max_row,max_col))

        power_pkg_table   = df.filter(regex=("^PW_PKG[0-9]*"))
        power_dram_table  = df.filter(regex=("^PW_DRAM[0-9]*"))

        power_row, power_col = power_pkg_table.shape
        pkg = power_col

        t   = df['TIME'].to_numpy()
        t_min = np.min(t)
        t_max = np.max(t)

        dram_energy = 0.0 
        pkg_energy = 0.0 
        for i in range(0, power_col):
            dram_energy += np.trapz(power_dram_table.iloc[:,i].to_numpy(),t)/1000.0
            pkg_energy += np.trapz(power_pkg_table.iloc[:,i].to_numpy(),t)/1000.0

        return dram_energy,pkg_energy,dram_energy+pkg_energy
    
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
            raise Exception(f'Failed compiling: { result.returncode }')

        mean_throughput = 0
        mean_energy = 0
        new_environment = dict(os.environ, KMP_AFFINITY='scatter')
        csv_path = 'temporary_csv_file.csv'
        for _ in range(num_evaluations):
            result_nrj = subprocess.run(['cpu_monitor_binary/releases/default/cpu_monitor.x','--csv',f'--csv-file={csv_path}','--quiet','--redirect',executable_path,
                                     str(self.problem_size_x),
                                     str(self.problem_size_y),
                                     str(self.problem_size_z),
                                     str(self.nthreads), '100',
                                     str(self.thrdblock_x),
                                     str(self.thrdblock_y),
                                     str(self.thrdblock_z)], 
                                     capture_output=True,
                                     env=new_environment)
            energy = self.csv_to_energy('temporary_csv_file.csv')[2]
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
                raise Exception(f'Failed executing: { result.returncode }')

            output = result.stdout
            m = re.search('throughput:\s+([\d\.]+)', str(output))
            throughput = m.group(1)
            try:
                mean_throughput += float(throughput)
                mean_energy += float(energy)
            except:
                raise ValueError('throughput not a float')
            if verbose:
                print(output)

        if delete_file:
            result = subprocess.run(['rm', executable_path])
            if result.returncode != 0:
                raise Exception(f'Failed deleting: { result.returncode }')

        mean_throughput = round(mean_throughput/num_evaluations, 2)
        mean_energy = round(mean_energy/num_evaluations, 2)
        self.calculated_cost = mean_throughput
        self.calculated_energy = mean_energy
        return mean_throughput/mean_energy
