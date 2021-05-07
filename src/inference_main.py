import argparse
import os

import numpy
import torch

import models
import ImageLoader

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
    dataset_group.add_argument("--num_workers",type=nonneg_int,default=0)
    
    model_group = arg_parser.add_argument_group("model")
    model_group.add_argument("--model",type=str,default="newModel")
    model_group.add_argument("--weight_file",type=str,required=True)
    
    arg_parser.add_argument("--out",type=str,required=True)
    arg_parser.add_argument("--best_matches",type=str,default=None)
    
    args = arg_parser.parse_args()
    #print(args)
    
    if args.num_workers != 0:
        print("num_workers != 0 currently causes memory leaks")
        arg_parser.exit(1)
    
    print("creating and loading model")
    model = models.create_model(args.model)
    model.load_state_dict( torch.load(args.weight_file) )
    
    print("Loading dataset")
    test_data = ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/",split="val")
    #test_data = ImageLoader.ImageFolderSubset(ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/"),list(range(1,100000,100)))
    
    inference_dataloader = torch.utils.data.DataLoader(test_data,shuffle=False,batch_size=200,num_workers=args.num_workers)
    
    embeddings = list()
    
    
    for batch_idx, (imgs, labels) in enumerate(inference_dataloader):
        if batch_idx % 10 == 0:
            print(batch_idx)
        embeddings.append(model(imgs).detach().numpy())#uses much less memory.
    
    embeddings = numpy.vstack(embeddings)
    numpy.savetxt(args.out, embeddings)
    #HDF5 would be much better.
    
    if args.best_matches is not None:
        with open(args.best_matches,"w") as outfile:
            #These three lines find the Euclidean distance between all vectors.
            b = numpy.dot(embeddings,embeddings.T)
            d = numpy.diag(b)
            c = numpy.power(d + d.reshape(-1,1) - 2.0*b, 0.5)
            numpy.fill_diagonal(c,c.max() + 10.0)#prevents becoming own best match. #self distance is always 0
            m = c.argmin(1)
            
            #TODO: This only gets the single best match. We need the best several.
            #Can accomplish with argsort.
            
            for i, (closest, (query_path, query_label)) in enumerate(zip(m, test_data.samples)):
                query_filename = os.path.basename(query_path)
                
                closest_path, closest_label = test_data.samples[closest]
                closest_filename = os.path.basename(closest_path)
                print(query_filename,query_label,closest_filename,closest_label,file=outfile)
    
    
if __name__ == "__main__":
    main()
