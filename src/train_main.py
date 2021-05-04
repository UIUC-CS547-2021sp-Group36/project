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
    
    wandb_group = arg_parser.add_argument_group("wandb","Arguments related to Weights and Biases")
    wandb_group.add_argument("--runname",type=str)
    wandb_group.add_argument("--resume",action="store_true")
    wandb_group.add_argument("--wandb-tags",type=str_list,default=[])
    
    dataset_group = arg_parser.add_argument_group("data")
    dataset_group.add_argument("--dataset","-d",metavar="TINY_IMAGENET_ROOT_DIRECTORY",type=str,default="/workspace/datasets/tiny-imagenet-200/")
    dataset_group.add_argument("--split",metavar="TRAIN_PROPORTION,CROSSVAL_PROPORTION",type=check_datasplit,default=[0.2,0.05,1.0-0.25])
    dataset_group.add_argument("--num_workers",type=nonneg_int,default=0)
    
    arg_parser.add_argument("--model",type=str,default="dummy")
    
    training_group = arg_parser.add_argument_group("training")
    training_group.add_argument("--batch-size",type=pos_int,default=200)
    training_group.add_argument("--epochs",metavar="N_epochs",type=int, nargs=1)
    training_group.add_argument("--loss",type=str,default="normed")
    training_group.add_argument("--margin","-g",type=nonneg_float)
    
    arg_parser.add_argument("--checkpoint","-c",type=int, default=50)
    
    args = arg_parser.parse_args()
    print(args)
    
    
    
    
    
if __name__ == "__main__":
    main()
