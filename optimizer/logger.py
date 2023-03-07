import os, sys
import time
import re
import shutil, glob
from matplotlib import pyplot as plt





class Logger():
    def __init__(self, process_id, logfile='lastrun.log', save_to_logfile=True, graphfile='graph.png', save_to_graphfile=True):
        self.Me = process_id

        self.log = open(logfile, "w") if save_to_logfile else None
        self.terminal = sys.stdout

        self.graphfile = graphfile
        self.save_to_graphfile = save_to_graphfile
    
    def write_msg(self, iteration_number, cost, compilation_flags, flair=None):
        # Example:
        # [15:12:58] [Me=0] [k=1] Cost=151.31 -O3 avx512 12 12 12 (New best!)
        current_time = time.strftime("%H:%M:%S", time.localtime())
        
        logstring = (f"[{current_time}]"
        f"\t[Me={self.Me}]"
        f"\t[k={iteration_number}]"
        f"\tCost={cost}"
        f"\t{compilation_flags}")
        if flair:
            logstring += f"\t({flair})"

        self.terminal.write(logstring + "\n")
        if self.log: self.log.write(logstring + "\n")
        if self.log: self.log.flush()

    def write_info(self, infostring):
        self.terminal.write(f"[info] [Me={self.Me}] " + infostring + "\n")
        if self.log: self.log.write(f"[info] [Me={self.Me}] " + infostring + "\n")
        if self.log: self.log.flush()

    def jumpline(self):
        self.terminal.write("\n")
        if self.log: self.log.write("\n")
        if self.log: self.log.flush()

    def write_raw(self, textstring):
        # Safe to use with strings starting with '\t'
        self.terminal.write(textstring + "\n")
        if self.log: self.log.write(textstring + "\n")
        if self.log: self.log.flush()

    def plot_graph(self, energy_path, index_path, max_steps):
        if self.save_to_graphfile: create_graph(energy_path, index_path, max_steps, self.graphfile)

    def __del__(self):
        if self.log: self.log.close()

def find_slurmfile(directory):
    candidates = glob.glob(directory + '/slurm-*.out')
    try:
        latest_slurmfile = candidates[0]
    except:
        raise FileNotFoundError(f"could not find a slurmfile in {directory}")
        
    for slurmfile in candidates[1:]:
        if os.path.getctime(slurmfile) > os.path.getctime(latest_slurmfile):
            latest_slurmfile = slurmfile
    return latest_slurmfile
        

def slurm_to_logfile(slurmfile, logfile):
    shutil.copy(slurmfile, logfile)

def log_to_list(logfile):
    """
    Converts a .log file to a list of dictionaries containing the log data (as strings).
    Example:
        data = log_to_list('hill-climbing.log')
        costs = [data[i]['Cost'] for i in range(len(data))]
        k = [data[i]['k'] for i in range(len(data))]
    """
    data = []
    with open(logfile, 'r') as f:
        for line in f:
            if line.startswith('[info]') or line.startswith('\t'):
                continue
            else:
                line_dict = {}
                for txt in line.split('\t'):
                    # remove brackets (when present)
                    txt = txt.translate({
                        ord('['): None,
                        ord(']'): None,
                    })
                    # time regex
                    m = re.search("/(?:[01]\d|2[0-3]):(?:[0-5]\d):(?:[0-5]\d)/", txt)
                    if m is not None:
                        line_dict['time'] = m.group(0)
                    else:
                    # process number, iteration number and cost regex
                        m = re.search("([A-Za-z]+)(=)(.*)", txt)
                        if m is not None:
                            if m.group(2) is not None:
                                if m.group(2) == '=':
                                    line_dict[m.group(1)] = m.group(3)
                    # TODO: compilation flags and flair
                if line_dict:
                    # if line_dict is not empty
                    data.append(line_dict)
    return data

def create_graph(energy_path, index_path, max_steps, graphfile):
    indexes = [*index_path, max_steps]
    x = range(max_steps)
    y = [0] * max_steps
    current_element_index = 0
    i = 0
    while i < max_steps:
        if i == indexes[current_element_index + 1]:
            current_element_index += 1
        y[i] = energy_path[current_element_index]
        i += 1

    # plt.rc('text', usetex=True)
    plt.rcParams.update({'font.family': 'serif'})
    # plt.rcParams.update({'font.size':16})
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(x, y, linewidth=2)
    ax1.grid(True)
    ax1.set_title('Cost function of best element up to each evaluation', fontsize=14)
    ax1.set_ylabel('Cost', fontsize=16)
    ax1.set_xlabel('Evaluation', fontsize=16)
    plt.tight_layout()
    plt.plot(x, y)
    plt.savefig(graphfile)