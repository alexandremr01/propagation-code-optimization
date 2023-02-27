import os
import subprocess
import threading

from solution import Solution
from multiprocessing import get_number_of_nodes

from algorithm import Algorithm

class SuccessiveDescents(Algorithm):
    def __init__(self, args) -> None:
        super().__init__(args)
    
    def run(self, steps):
        file_name = str(threading.get_ident())
        file_name_with_ext = f'launch_{file_name}.sh'
        with open(file_name_with_ext, "w") as script_file:
            script = self.get_script(4, 
                f'python3 src/optimizer.py --steps {steps} --algorithm hill_climbing --phase run --seed {self.args.seed}')
            script_file.write(script)

        cmd = f"sbatch -p cpu_prod --exclusive -N 4 -n 128 --qos=16nodespu {file_name_with_ext}" 
        print("Executed command: " + cmd)
        print("---->")
        res = subprocess.run(cmd,shell=True, env=os.environ)
    
    def get_script(self, process_number, command):
        return f"""#!/bin/sh
#SBATCH --time=15
echo "================================="
/usr/bin/mpirun -np {process_number} -map-by ppr:1:node:PE=16 -rank-by node {command}
echo "================================="
"""