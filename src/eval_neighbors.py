import argparse
import os

import warnings

import numpy
import torch

import models
import data.ImageLoader as ImageLoader
import LossFunction

import sklearn.neighbors

def final_accuracy(model, a_dataloader, triplet_accuracy=None, device=None, divide_at_end=True):
    if args.use_device is not None:
        model = model.to(device)
    model.eval()
    
    if triplet_accuracy is None:
        triplet_accuracy = LossFunction.TripletAccuracy()
    try:
        if args.use_device is not None:
            triplet_accuracy = triplet_accuracy.to(device)
    except:
        warnings.warn("Could not move accuracy function to selected device...")
    
    total_correct = 0.0
    total_seen = 0
    with torch.no_grad():
        for batch_idx, ((Qs,Ps,Ns),l) in enumerate(a_dataloader):
            
            if device is not None:
                Qs = Qs.to(device)
                Ps = Ps.to(device)
                Ns = Ns.to(device)
            
            Q_emb = model(Qs).detach()
            P_emb = model(Ps).detach()
            N_emb = model(Ns).detach()
            
            total_correct += float(triplet_accuracy(Q_emb, P_emb, N_emb))
            total_seen += int(len(l))
    
    if divide_at_end:
        return total_correct / float(total_seen)
    else:
        return total_correct, total_seen
    

def embed_using_model(model, a_dataloader, device=None, normalize=False):
    if device is not None:
        model = model.to(device)
    
    embeddings = list()
    
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(a_dataloader):
            if batch_idx % 10 == 0:
                print(batch_idx)
            
            if device is not None:
                imgs = imgs.to(device)
            
            some_emb = model(imgs).detach()
            
            if normalize:
                #for a while, we tried putting output normalization into the loss function
                some_emb = torch.nn.functional.normalize(some_emb).detach()
                
            embeddings.append(some_emb.detach().cpu().numpy())#uses much less memory.
    
    embeddings = numpy.vstack(embeddings)
    return embeddings

from util.cli import *

#MAIN
arg_parser = argparse.ArgumentParser(description='Train an image similarity vector embedding')
arg_parser.add_argument("--verbose","-v",action="store_true")
arg_parser.add_argument("--seed",type=int,default=1234)
arg_parser.add_argument("--config",type=load_yaml_file,default=None,help="Use the config from a run to automatically choose model and batch_size.")

dataset_group = arg_parser.add_argument_group("data")
dataset_group.add_argument("--dataset","-d",metavar="TINY_IMAGENET_ROOT_DIRECTORY",type=str,default="/workspace/datasets/tiny-imagenet-200/")
dataset_group.add_argument("--train_split",metavar="DATABASE_PROPRTION",type=check_datasplit,default=[0.1,0.1,0.8],help="Don't use all the data.")
dataset_group.add_argument("--test_split",metavar="QUERY_PROPORTION",type=check_datasplit,default=[0.1,0.1,0.8],help="Don't use all the data.")
dataset_group.add_argument("--batch_size",type=int,default=200)
dataset_group.add_argument("--num_workers",type=nonneg_int,default=0)

model_group = arg_parser.add_argument_group("model")
model_group.add_argument("--model",type=str,default="LowDNewModel")
model_group.add_argument("--weight_file",type=str,default=None)

#arg_parser.add_argument("--out",type=str,required=True)
#arg_parser.add_argument("--best_matches",type=str,default=None)
#arg_parser.add_argument("--n_best",type=int,default=10)
#arg_parser.add_argument("--n_worst",type=int,default=10)

args = arg_parser.parse_args()
if args.config is not None:
    args.model = args.config["model"]["value"]
    args.batch_size = args.config["batch_size"]["value"]

if args.num_workers != 0:
    print("num_workers != 0 currently causes memory leaks")
    


#CUDA
#TODO: Dependent upon cuda availability
args.use_cuda = False
args.use_device = "cpu"
if torch.cuda.is_available():
    print("CUDA is available, so we're going to try to use that!")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args.use_cuda = True
    args.use_device = "cuda:0"


#=========== MODEL ==============
print("creating and loading model")
model = models.create_model(args.model)
if args.weight_file is not None:
    model.load_state_dict( torch.load(args.weight_file,map_location=torch.device("cpu")) )
else:
    print("Warning, no weights loded. Predicting with default/initial weights.")
model.eval()

if args.use_cuda:
    model = model.to(args.use_device)


#============= DATA ==============

print("Loading datasets")
import random
random.seed(args.seed)

## TRAIN DATA
all_train = ImageLoader.load_imagefolder(args.dataset,split="train")
database_dataset, _, _ = ImageLoader.split_imagefolder(all_train, args.train_split)

## TEST DATA
#load the crossval split of TinyImageNet (which we are using as a test split)
all_test = ImageLoader.load_imagefolder(args.dataset,split="val")
query_dataset, _, _ = ImageLoader.split_imagefolder(all_test, args.test_split)

#================ TRIPLET ACCURACY ===========================
train_tsdl = ImageLoader.TripletSamplingDataLoader(database_dataset,
                                    shuffle=False,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)
test_tsdl = ImageLoader.TripletSamplingDataLoader(query_dataset,
                                    shuffle=False,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

train_accuracy =  final_accuracy(model, train_tsdl, device=args.use_device)
print("Accuracy on (subsample of) training triplets: {:.5f}".format(train_accuracy))

test_accuracy  =  final_accuracy(model, test_tsdl, device=args.use_device)
print("Accuracy on (subsample of) test triplets: {:.5f}".format(test_accuracy))


#================ EMBEDDINGS AND KNN =========================

db_dataloader    = torch.utils.data.DataLoader(database_dataset,shuffle=False,batch_size=args.batch_size,num_workers=args.num_workers)
query_dataloader = torch.utils.data.DataLoader(query_dataset,shuffle=False,batch_size=args.batch_size,num_workers=args.num_workers)

db_embeddings    = embed_using_model(model, db_dataloader, device=args.use_device)
query_embeddings = embed_using_model(model, query_dataloader, device=args.use_device)
#numpy.savetxt(args.out, embeddings)
#HDF5 would be much better.

#============= SAVE EMBEDDINGS IF SO DESIRED ===========
#============= OR SKIP ALL ABOVE AND LOAD IF YOU WANT =======

knn_database   = sklearn.neighbors.NearestNeighbors(n_neighbors=200,radius=10.0)
knn_database.fit(db_embeddings)

dists, indices = knn_database.kneighbors(query_embeddings,5,return_distance=True)

query_classes  = numpy.array(query_dataset.targets)
hit_classes    = numpy.array(numpy.array(database_dataset.targets)[indices])

accuracy = (query_classes == hit_classes[:,0]).mean()
print("top accuracy ('knn-1'): {}".format(accuracy) )

dharawat_accuracy = ((query_classes.reshape(-1,1) == hit_classes).sum(1) > 0).mean()
print("dharawat accuracy ('knn-30'): {}".format(dharawat_accuracy))
    
