import os
import subprocess
import threading

from optimizer.solution import Solution

from optimizer.algorithms import Algorithm
    
def deploy_kangaroo(args, script_name, logger):
    file_name = str(threading.get_ident())
    file_name_with_ext = f'launch_{file_name}.sh'

    args.phase = 'run'
    formatted_new_args = print_args(args)

    with open(file_name_with_ext, "w") as script_file:
        script = get_script(4, 
            f'python3 -m optimizer.main {formatted_new_args}')
        script_file.write(script)

    cmd = f"sbatch -p cpu_prod --reservation st76intelprod --exclusive -N 4 -n 128 --qos=16nodespu {file_name_with_ext}"
    logger.write_info("Executed command: " + cmd)
    logger.write_info("Waiting for batch job completion...")
    res = subprocess.run(cmd,shell=True, env=os.environ)

def deploy_single(args, script_name, logger):
    args.phase = 'run'
    formatted_new_args = print_args(args)
    cmd = f'/usr/bin/mpirun -np 1 -map-by ppr:1:node:PE=16 python3 -m optimizer.main {formatted_new_args}'
    logger.write_info("Executed command: " + cmd)
    # print("---->")
    subprocess.run(cmd, shell=True, env=os.environ)
    
def get_script(process_number, command):
    return f"""#!/bin/sh
#SBATCH --time=15
echo "================================="
/usr/bin/mpirun -np {process_number} -map-by ppr:1:node:PE=16 -rank-by node {command}
echo "================================="
"""

def print_args(args):
    s = ''
    for k, v in sorted(vars(args).items()):
        if type(v) == bool:
            s += f'--{k} ' if v else ''
        elif type(v) == list:
            str_v = [str(x) for x in v]
            str_v = ' '.join(str_v)
            s += f'--{k} {str_v} '
        elif k=='hparams':
            s += f'--{k} \'{v}\' '
        else:
            s += f'--{k} {v} '
    return s
