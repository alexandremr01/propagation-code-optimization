import subprocess
import os
import re 
import numpy as np

def execute_with_affinity(affinity=None, nthreads=16, num_evaluations=20):
    print('Affinity: ', affinity, ' Threads: ', nthreads)
    file_name_with_ext = f'test_affinity.exe'
    executable_path = f'iso3dfd-st7/bin/{file_name_with_ext}'

    new_environment = dict(os.environ, CONFIG_EXE_NAME=file_name_with_ext)
    if affinity is not None:
        new_environment['KMP_AFFINITY'] = affinity
    result = subprocess.run(['make', '-C', 'iso3dfd-st7', f'Olevel=-O2', f'simd=avx2', 'last'],
                            stdout=subprocess.DEVNULL,
                            env=new_environment)
    if result.returncode != 0:
        raise Exception(f'Failed compiling: { result.returncode }')

    throughputs = [ ]
    arguments = '256 256 256 ' + str(nthreads) + ' 100 256 2 4'
    parameters = [executable_path] + arguments.split(' ')
    for _ in range(num_evaluations):
        result = subprocess.run(parameters, capture_output=True, env=new_environment)
        if result.returncode != 0:
            raise Exception(f'Failed executing: { result.returncode }')

        output = result.stdout
        m = re.search('throughput:\s+([\d\.]+)', str(output))
        throughput = m.group(1)
        try:
            throughputs.append(float(throughput))
        except:
            raise ValueError('throughput not a float')
        
    print('Throughputs: ', throughputs)
    print('Mean:', np.average(throughputs))
    print('Std:', np.std(throughputs))
    print('Std/Mean %: ', 100*np.std(throughputs) / np.average(throughputs), '\n\n')

list_nthreads = [16, 32]
affinities = [None, 'compact', 'scatter', 'balanced']
for nthreads in list_nthreads:
    for affinity in affinities:
        execute_with_affinity(affinity, nthreads)