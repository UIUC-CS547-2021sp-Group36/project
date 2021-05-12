import argparse

import yaml

def load_yaml_file(fn):
    if fn is None:
        return None
    with open(fn,"r") as stream:
        return yaml.load(stream,Loader=yaml.Loader)

def pos_int(i):
    ival = int(i)
    if ival <= 0:
        raise argparse.ArgumentTypeError("{} is not a positive integer".format(i))
    return ival

def nonneg_int(i):
    ival = int(i)
    if ival < 0:
        raise argparse.ArgumentTypeError("{} is not a non-negative integer".format(i))
    return ival

def nonneg_float(f):
    fval = float(f)
    if fval < 0.0:
        raise argparse.ArgumentTypeError("{} is not a non-negative float.".format(f))
    return fval
    
class float_in_range(object):
    def __init__(self,min=float("-inf"),max=float("-inf")):
        self.min = min
        self.max = max
        self.range = "[" if self.min is not float("-inf") else "("
        self.range += "{:.4f}".format(self.min)
        self.range += ","
        self.range += "{:.4f}".format(self.max)
        self.range = "}" if self.max is not float("inf") else ")"
        
    
    def __call__(self,v):
        v = float(v)
        if not self.min <= v <= self.max:
            raise argparse.ArgumentTypeError("{:.5f} is not in the range {}".format(v,self.range))
        return v
    
def str_list(s):
    return s.split(",")
    
def check_datasplit(in_str):
    splits = list(map(float,in_str.split(",")))
    
    if len(splits) == 1:
        splits.append(1.0 - sum(splits))
    if len(splits) != 2:
        raise argparse.ArgumentTypeError("You must provide one or two values for splits, comma separated. Their sum must be <= 1.0")
    if not 0.0 < sum(splits) <= 1.0:
        raise argparse.ArgumentTypeError("The sum of values must be on the interval (0.0,1.0], you provided {} which sum to {}".format(
                                            in_str,
                                            sum(splits)
        ))
    if len(splits) == 2:
        splits.append(1.0-sum(splits))
    if len(splits) != 3:
        raise argparse.ArgumentTypeError("Some kind of problem happened. Sorry.")
    return splits
