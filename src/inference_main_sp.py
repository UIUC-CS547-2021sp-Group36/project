import argparse
import os

import numpy
import torch

import models
import data.ImageLoader as ImageLoader

import LossFunction

def pairwise_distances(matrix_of_rows, epsilon=1.0e-6):
    """
    Computes the pairwise_distances between all pairs of rows.
    
    Returns a square matrix. Diagonal will be zeros.
    """
    
    #TODO: replace with scipy.spatial.distance.pdist
    #DOCS https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
    
    b = numpy.dot(matrix_of_rows,matrix_of_rows.T)
    d = numpy.diag(b)
    c_input = d + d.reshape(-1,1) - 2.0*b
    #numpy doesn't like negative inputs to power with fractional power.
    #UGLY # HACK
    c_input += epsilon
    c = numpy.power(c_input, 0.5)
    return c

def nonneg_int(i):
    ival = int(i)
    if ival < 0:
        raise argparse.ArgumentTypeError("{} is not a non-negative integer".format(i))
    return ival
    
def main():
    arg_parser = argparse.ArgumentParser(description='Train an image similarity vector embedding')
    arg_parser.add_argument("--verbose","-v",action="store_true")
    
    dataset_group = arg_parser.add_argument_group("data")
    dataset_group.add_argument("--dataset","-d",metavar="TINY_IMAGENET_ROOT_DIRECTORY",type=str,default="/workspace/datasets/tiny-imagenet-200/")
    dataset_group.add_argument("--batch_size",type=int,default=200)
    dataset_group.add_argument("--num_workers",type=nonneg_int,default=0)
    
    model_group = arg_parser.add_argument_group("model")
    model_group.add_argument("--model",type=str,default="LowDNewModel")
    model_group.add_argument("--weight_file",type=str,default=None)
    
    arg_parser.add_argument("--out",type=str,required=True)
    arg_parser.add_argument("--best_matches",type=str,default=None)
    arg_parser.add_argument("--n_best",type=int,default=10)
    arg_parser.add_argument("--n_worst",type=int,default=10)
    
    args = arg_parser.parse_args()
    #print(args)
    
    if args.num_workers != 0:
        print("num_workers != 0 currently causes memory leaks")
        arg_parser.exit(1)
    
    print("creating and loading model")
    model = models.create_model(args.model)
    if args.weight_file is not None:
        model.load_state_dict( torch.load(args.weight_file,map_location=torch.device("cpu")) )
    else:
        print("Warning, no weights loded. Predicting with default/initial weights.")
    model.eval()
    
    print("Loading dataset")
    #load the crossval split of TinyImageNet (which we are using as a test split)
    test_data = ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/",split="val")
    #load from the training data.
    #test_data = ImageLoader.ImageFolderSubset(ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/"),list(range(1,100000,100)))
    
    inference_dataloader = ImageLoader.TripletSamplingDataLoader(test_data,shuffle=False,batch_size=args.batch_size,num_workers=args.num_workers)
    
    
    accuracy_function = LossFunction.TripletAccuracy()
    total_acc = 0.0
    total_N = 0
    
    with torch.no_grad():
        for batch_idx, ((Qs,Ps,Ns),l) in enumerate(inference_dataloader):
            if batch_idx % 10 == 0:
                print(batch_idx)
            
            Q_emb = model(Qs).detach()
            P_emb = model(Ps).detach()
            N_emb = model(Ns).detach()
            
            total_acc += float(accuracy_function(Q_emb, P_emb, N_emb))
            total_N += int(len(l))
    
    total_acc /= float(total_N)
    
    print("Total accuracy on test data: {}".format(total_acc))
    
if __name__ == "__main__":
    main()
