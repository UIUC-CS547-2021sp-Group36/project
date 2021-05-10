import argparse

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
    
def str_list(s):
    return s.split(",")
    
def check_datasplit(in_str):
    splits = list(map(float,in_str.split(",")))
    if len(splits) != 2:
        raise argparse.ArgumentTypeError("You must provide two values for splits, comma separated. Their sum must be <= 1.0")
    if not 0.0 < sum(splits) <= 1.0:
        raise argparse.ArgumentTypeError("The sum of values must be on the interval (0.0,1.0], you provided {} which sum to {}".format(
                                            in_str,
                                            sum(splits)
        ))
    splits.append(1.0-sum(splits))
    return splits
