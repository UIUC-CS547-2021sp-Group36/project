import argparse
import os

import numpy
import torch

import models
import data.ImageLoader as ImageLoader

def nonneg_int(i):
    ival = int(i)
    if ival < 0:
        raise argparse.ArgumentTypeError("{} is not a non-negative integer".format(i))
    return ival
    
#def main():
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Train an image similarity vector embedding')
    arg_parser.add_argument("--verbose","-v",action="store_true")
    
    dataset_group = arg_parser.add_argument_group("data")
    dataset_group.add_argument("--dataset","-d",metavar="TINY_IMAGENET_ROOT_DIRECTORY",type=str,default="/workspace/datasets/tiny-imagenet-200/")
    dataset_group.add_argument("--num_workers",type=nonneg_int,default=0)
    
    model_group = arg_parser.add_argument_group("model")
    model_group.add_argument("--model",type=str,default="LowDNewModel")
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
    if args.weight_file != "-": #Default weights
        model.load_state_dict( torch.load(args.weight_file) )
    
    print("Loading dataset")
    test_data = ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/",split="val")
    #test_data = ImageLoader.ImageFolderSubset(ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/"),list(range(1,100000,100)))
    
    inference_dataloader = torch.utils.data.DataLoader(test_data,shuffle=False,batch_size=200,num_workers=args.num_workers)
    
    embeddings = list()
    
    for batch_idx, (imgs, labels) in enumerate(inference_dataloader):
        if batch_idx % 10 == 0:
            print(batch_idx)
        some_emb = model(imgs).detach()
        #some_emb = torch.nn.functional.normalize(some_emb).detach()
        embeddings.append(some_emb.detach().numpy().argmax(1).reshape(-1,1))#uses much less memory.
    
    embeddings = numpy.vstack(embeddings)
    embeddings = numpy.hstack([embeddings, numpy.array(test_data.targets).reshape(-1,1)])
    numpy.savetxt(args.out, embeddings,fmt="%i")
    
#if __name__ == "__main__":
#    main()
