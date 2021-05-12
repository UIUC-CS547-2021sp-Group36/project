import argparse
import os
import sys

import warnings

import numpy
import torch

import models
import data.ImageLoader as ImageLoader
import LossFunction

import sklearn.neighbors
import scipy.spatial.distance

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
                print(batch_idx,file=sys.stderr)
            
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
    print("num_workers != 0 currently causes memory leaks",file=sys.stderr)
    


#CUDA
#TODO: Dependent upon cuda availability
args.use_cuda = False
args.use_device = "cpu"
if torch.cuda.is_available():
    print("CUDA is available, so we're going to try to use that!",file=sys.stderr)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args.use_cuda = True
    args.use_device = "cuda:0"


#=========== MODEL ==============
print("creating and loading model",file=sys.stderr)
model = models.create_model(args.model)
if args.weight_file is not None:
    model.load_state_dict( torch.load(args.weight_file,map_location=torch.device("cpu")) )
else:
    print("Warning, no weights loded. Predicting with default/initial weights.",file=sys.stderr)
model.eval()

if args.use_cuda:
    model = model.to(args.use_device)


#============= DATA ==============

print("Loading datasets",file=sys.stderr)
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

#============= Select N=5 random query images and find the best and worst M=10 for them.
N_for_display = 5
M_best_worst = 10
N_train = db_embeddings.shape[0]
N_test  = query_embeddings.shape[0]

random_queries = set()
assert N_test > N_for_display, "Can't choose {} unique items from a set of size {}!".format(N_for_display, N_test)
while len(random_queries) < N_for_display:
    random_queries.add(numpy.random.randint(0,N_test))
random_queries = list(random_queries)
random_queries.sort()
random_queries = numpy.array(random_queries)

display_query_embeddings = query_embeddings[random_queries]

#Can't use KNN to find the _worst_ matches. Must compare all.
display_distances = scipy.spatial.distance.cdist(display_query_embeddings,db_embeddings,metric='euclidean')
sorted_keys = display_distances.argsort(1)
display_best_M  = sorted_keys[:,:M_best_worst]
display_worst_M = sorted_keys[:,-M_best_worst:]

#get filename and label for all the matches.
query_paths_labels = [query_dataset.imgs[i] for i in random_queries]
best_paths_labels  = [[database_dataset.imgs[i] for i in j] for j in display_best_M]
worst_paths_labels = [[database_dataset.imgs[i] for i in j] for j in display_worst_M]

#Basenames of all the filenames
#We can look up the classname if we need, but we'll save it just for convenience.
query_paths_labels = list(map(lambda x:(os.path.basename(x[0]),x[1]), query_paths_labels))

#The training images have their classname in the image name. It7s inconvenient in this file.
#best_paths  = [list(map(lambda x:os.path.basename(x[0]), j)) for j in best_paths_labels]
best_paths  = [list(map(lambda x:x[0], j)) for j in best_paths_labels]
best_labels = [list(map(lambda x:x[1], j)) for j in best_paths_labels]

#worst_paths  = [list(map(lambda x:os.path.basename(x[0]), j)) for j in worst_paths_labels]
worst_paths  = [list(map(lambda x:x[0], j)) for j in worst_paths_labels]
worst_labels  = [list(map(lambda x:x[1], j)) for j in worst_paths_labels]

#TODO: Just save everything in HDF5 or some kind of structured storage.
numpy.savetxt("queries.txt",query_paths_labels,fmt="%s")
numpy.savetxt("queries_emb.txt",display_query_embeddings)

numpy.savetxt("best_paths_for_display.txt",numpy.array(best_paths).reshape(N_for_display,-1),fmt="%s")
numpy.savetxt("best_labels_for_display.txt",numpy.array(best_labels).reshape(N_for_display,-1),fmt="%s")
numpy.savez_compressed("best_embeddings.npz",db_embeddings[display_best_M])
#TODO: this could be done with numpy indices if I could figure them out.
#numpy.savetxt("best_distances.txt",display_distances[display_best_M])
numpy.savetxt("best_distances.txt",numpy.array([display_distances[i,display_best_M[i]] for i in range(display_distances.shape[0])]))


numpy.savetxt("worst_paths_for_display.txt",numpy.array(worst_paths).reshape(N_for_display,-1),fmt="%s")
numpy.savetxt("worst_labels_for_display.txt",numpy.array(worst_labels).reshape(N_for_display,-1),fmt="%s")
numpy.savez_compressed("worst_embeddings.npz",db_embeddings[display_worst_M])
#numpy.savetxt("worst_distances.txt",display_distances[display_worst_M])
numpy.savetxt("worst_distances.txt",numpy.array([display_distances[i,display_worst_M[i]] for i in range(display_distances.shape[0])]))
