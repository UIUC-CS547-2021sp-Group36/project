import argparse
import os

import numpy
import torch

import models
import data.ImageLoader as ImageLoader

import sklearn.neighbors

def embed_using_model(model, a_dataloader):
    embeddings = list()
    
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(a_dataloader):
            if batch_idx % 10 == 0:
                print(batch_idx)
            some_emb = model(imgs).detach()
            some_emb = torch.nn.functional.normalize(some_emb).detach()
            embeddings.append(some_emb.detach().numpy())#uses much less memory.
    
    embeddings = numpy.vstack(embeddings)
    return embeddings

from util.cli import *

#MAIN
arg_parser = argparse.ArgumentParser(description='Train an image similarity vector embedding')
arg_parser.add_argument("--verbose","-v",action="store_true")
arg_parser.add_argument("--seed",type=int,default=1234)

dataset_group = arg_parser.add_argument_group("data")
dataset_group.add_argument("--dataset","-d",metavar="TINY_IMAGENET_ROOT_DIRECTORY",type=str,default="/workspace/datasets/tiny-imagenet-200/")
dataset_group.add_argument("--train_split",metavar="DATABASE_PROPRTION",type=float_in_range(0.0,1.0),default=0.1,help="Don't use all the data.")
dataset_group.add_argument("--test_split",metavar="QUERY_PROPORTION",type=float_in_range(0.0,1.0),default=0.05,help="Don't use all the data.")
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
args.train_split = [args.train_split,(1.0-args.train_split)/2.0, (1.0-args.train_split)/2.0]
args.test_split  = [args.test_split,(1.0-args.test_split)/2.0, (1.0-args.test_split)/2.0]
#print(args)

if args.num_workers != 0:
    print("num_workers != 0 currently causes memory leaks")
    arg_parser.exit(1)
    


#CUDA
#TODO: Dependent upon cuda availability
use_cuda = False
use_device = "cpu"
if torch.cuda.is_available():
    print("CUDA is available, so we're going to try to use that!")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    use_cuda = True
    use_device = "cuda:0"


#=========== MODEL ==============
print("creating and loading model")
model = models.create_model(args.model)
if args.weight_file is not None:
    model.load_state_dict( torch.load(args.weight_file,map_location=torch.device("cpu")) )
else:
    print("Warning, no weights loded. Predicting with default/initial weights.")
model.eval()

if use_cuda:
    model = model.to(use_device)


#============= DATA ==============

print("Loading datasets")
import random
random.seed(args.seed)

all_train = ImageLoader.load_imagefolder(args.dataset,split="train")
database_dataset, _, _ = ImageLoader.split_imagefolder(all_train, args.train_split)


#load the crossval split of TinyImageNet (which we are using as a test split)
all_test = ImageLoader.load_imagefolder(args.dataset,split="val")
query_dataset, _, _ = ImageLoader.split_imagefolder(all_test, args.test_split)
#load from the training data.
#test_data = ImageLoader.ImageFolderSubset(ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/"),list(range(1,100000,100)))

db_dataloader    = torch.utils.data.DataLoader(database_dataset,shuffle=False,batch_size=args.batch_size,num_workers=args.num_workers)
query_dataloader = torch.utils.data.DataLoader(query_dataset,shuffle=False,batch_size=args.batch_size,num_workers=args.num_workers)

db_embeddings    = embed_using_model(model, db_dataloader)
query_embeddings = embed_using_model(model, query_dataloader)
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
print("top accuracy: {}".format(accuracy) )

dharawat_accuracy = ((query_classes.reshape(-1,1) == hit_classes).sum(1) > 0).mean()
print("dharawat accuracy ('knn-30'): {}".format(dharawat_accuracy))
    
