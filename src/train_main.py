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
    
def main():
    arg_parser = argparse.ArgumentParser(description='Train an image similarity vector embedding')
    arg_parser.add_argument("--verbose","-v",action="store_true")
    
    
    arg_parser.add_argument("--runname",type=str, nargs="+")
    arg_parser.add_argument("--resume",action="store_true")
    arg_parser.add_argument("--wandb-tags",type=str_list, nargs="+")
    
    arg_parser.add_argument("--dataset","-d",type=str,default="/workspace/datasets/tiny-imagenet-200/")
    arg_parser.add_argument("--split",type=check_datasplit,default=[0.2,0.05,1.0-0.25])
    arg_parser.add_argument("--num_workers",type=nonneg_int,default=0)
    
    arg_parser.add_argument("--model",type=str,default="dummy")
    
    
    arg_parser.add_argument("--batch-size",type=int,default=200)
    arg_parser.add_argument("--epochs",metavar="N_epochs",type=int, nargs=1)
    arg_parser.add_argument("--checkpoint","-c",type=int, default=50)
    
    arg_parser.add_argument("--loss",type=str,nargs="+",default="normed")
    arg_parser.add_argument("--margin","-g",type=nonneg_float)
    
    args = arg_parser.parse_args()
    print(args)
    
    
    
    
    
if __name__ == "__main__":
    main()
