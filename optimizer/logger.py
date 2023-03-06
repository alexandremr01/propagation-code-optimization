import os, sys
import time
import re

class Logger():
    def __init__(self, process_id, logfile):
        self.Me = process_id

        self.terminal = sys.stdout
        self.log = open(logfile, "w")
    
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
        self.log.write(logstring + "\n")

    def write_info(self, infostring):
        self.terminal.write("[info] " + infostring + "\n")
        self.log.write("[info] " + infostring + "\n")

    def jumpline(self):
        self.terminal.write("\n")
        self.log.write("\n")

    def write_raw(self, textstring):
        # Safe to use with strings starting with '\t'
        self.terminal.write(textstring + "\n")
        self.log.write(textstring + "\n")

    def __del__(self):
        self.log.close()

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