import argparse
import os

import numpy
import torch

import models
import data.ImageLoader as ImageLoader

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
    
    print("Loading dataset")
    #load the crossval split of TinyImageNet (which we are using as a test split)
    test_data = ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/",split="val")
    #load from the training data.
    #test_data = ImageLoader.ImageFolderSubset(ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/"),list(range(1,100000,100)))
    
    inference_dataloader = torch.utils.data.DataLoader(test_data,shuffle=False,batch_size=200,num_workers=args.num_workers)
    
    embeddings = list()
    
    
    for batch_idx, (imgs, labels) in enumerate(inference_dataloader):
        if batch_idx % 10 == 0:
            print(batch_idx)
        some_emb = model(imgs).detach()
        some_emb = torch.nn.functional.normalize(some_emb).detach()
        embeddings.append(some_emb.detach().numpy())#uses much less memory.
    
    embeddings = numpy.vstack(embeddings)
    numpy.savetxt(args.out, embeddings)
    #HDF5 would be much better.
    
    if args.best_matches is not None:
        with open(args.best_matches,"w") as matches_outfile:
            #These three lines find the Euclidean distance between all vectors.
            c = pairwise_distances(embeddings)
            numpy.fill_diagonal(c,c.max() + 10.0)#prevents becoming own best match. #self distance is always 0
            
            m = c.argsort(1)
            
            m_best  = m[:,:args.n_best]
            m_worst = m[:,-args.n_worst:]
            
            #TODO: This only gets the single best match. We need the best several.
            #Can accomplish with argsort.
            
            #TODO: Write a header line
            header_line = "#QUERY\tQ_LABEL\t"
            header_line += "\t".join(["BEST_{}\tBEST_LABEL_{}".format(i,i) for i in range(args.n_best)])
            header_line += "\t" + "\t".join(["WORST_{}\tWORST_LABEL_{}".format(i,i) for i in range(args.n_worst)][::-1])
            print(header_line,file=matches_outfile)
            
            for i, (closest, furthest, (query_path, query_label)) in enumerate(zip(m_best, m_worst, test_data.samples)):
                query_filename = os.path.basename(query_path)
                
                bn_label = lambda samp:(os.path.basename(samp[0]),samp[1])
                closest_fns_label = [bn_label(test_data.samples[c]) for c in closest]
                furthest_fns_label = [bn_label(test_data.samples[f]) for f in furthest]
                
                output_chunks = "\t".join(["{}\t{}".format(fn,l) for fn,l in closest_fns_label])
                output_chunks += "\t" + "\t".join(["{}\t{}".format(fn,l) for fn,l in furthest_fns_label])
                
                print(query_filename,query_label,output_chunks,
                        sep="\t",
                        file=matches_outfile
                      )
    
    
if __name__ == "__main__":
    main()
