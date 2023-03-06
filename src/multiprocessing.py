import subprocess

def get_number_of_nodes( ):
  result_srun = subprocess.run(["srun", "hostname"], capture_output=True)
  srun_output = str(result_srun.stdout)
  hosts = set(srun_output.split('\n'))
  return len(hosts)
