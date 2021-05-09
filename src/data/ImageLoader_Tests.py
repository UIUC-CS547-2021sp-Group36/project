#This is just initial experiment code for now.

from .ImageLoader import *

class DebugTripletSamplingDataLoader(TripletSamplingDataLoader):
    """
    Subclass only for debugging.
    
    In our case, collate_fn needs to be a member function, so we need this class.
    """
    def __init__(self, dataset:torchvision.datasets.ImageFolder,
                            batch_size=20,
                            shuffle=True,
                            num_workers: int = 0):
        super(DebugTripletSamplingDataLoader, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False)
    
    
    def collate_fn(self, somedata):
        #wi = torch.utils.data.get_worker_info()
        
        query_tensor = [i[0] for i in somedata]
        labels = [i[1] for i in somedata]
        
        positive_image_indices = [self.label_index.sample_item(label=l) for l in labels]
        negative_image_indices = [self.label_index.sample_item(exclude=l) for l in labels]
        
        positive_image_tensor = [self.dataset[i][0] for i in positive_image_indices]
        negative_image_tensor = [self.dataset[i][0] for i in negative_image_indices]
        
        return (query_tensor, positive_image_tensor, negative_image_tensor), torch.IntTensor(labels).detach(), positive_image_indices, negative_image_indices

if __name__ == "__main__":
    
    import torch
    print("load data")
    all_train = load_imagefolder("/workspace/datasets/tiny-imagenet-200",
                                    transform=lambda x:x,
                                    is_valid_file=lambda x: x.rsplit(".",1)[-1] == "JPEG"
                                )
    
    
    print("index data")
    an_index = ImageFolderLabelIndex(all_train)
    
    print("done indexing data")
    
    if False:
        print("Example sampling")
        for i in range(20):
            l = an_index.sample_label(weighted_classes=False)
            i_query = an_index.sample_item(label=l)
            i_pos = an_index.sample_item(label=l)
            i_neg = an_index.sample_item(exclude=l)
        
            print(l,i_query, i_pos, i_neg)
        
    tsdl = TripletSamplingDataLoader(all_train,batch_size=20, num_workers=0)
    
    if False:
        import torchvision.models as models
        resnet18 = models.resnet18(pretrained=True)
        
        for i, ((q,p,n),l) in enumerate(tsdl):
            print(q.is_pinned())
            print(p.is_pinned())
            print(n.is_pinned())
            print("batch ", i, l.tolist())
            
            q_emb = resnet18(q)
            
            print(q_emb)
            
            if i == 3:
                break
    
    if True:
        subset_test = ImageFolderSubset(all_train,[1,2,3,670])
        print(subset_test.targets)
    
    
    #===================== RANDOM SAMPLING =========================
    if True:
        import os
        print("Tests of random sampling")
        test_split_size = 0.9
        test_shuffle = True
        test_batch_size = 20
        test_num_workers = 0
        
        #Looks like we can't monkey patch the object after creation.
        #Must monkey patch the class
        __original_getitem = TinyImageNet.__getitem__
        def raw_path(self, i):
            return self.imgs[i]
        TinyImageNet.__getitem__ = raw_path
        
        no_transform_dataset = load_imagefolder("/workspace/datasets/tiny-imagenet-200",
                                        transform=lambda x:x,
                                        is_valid_file=lambda x: x.rsplit(".",1)[-1] == "JPEG"
                                    )
        
        data_use_for_test,_ = split_imagefolder(no_transform_dataset,[test_split_size,1.0-test_split_size])
        
        tsdl = DebugTripletSamplingDataLoader(data_use_for_test,shuffle=test_shuffle, batch_size=test_batch_size, num_workers=test_num_workers)
        for (qs,ps,ns), l, p_index, n_index in tsdl:
            
            #from the filenames
            for one_q, one_p, one_n in zip(qs,ps,ns):
                one_q = os.path.basename(one_q).split("_",1)[0]
                one_p = os.path.basename(one_p).split("_",1)[0]
                one_n = os.path.basename(one_n).split("_",1)[0]
                assert one_q == one_p and one_q != one_n , "The classes of Q and P must match, and those of Q and N must differ"
                
                
            
            #From the labels
            for one_l, one_p_index, one_n_index in zip(l,p_index, n_index):
                assert one_l == tsdl.dataset[one_p_index][1], "Positive labels do not match"
                assert one_l != tsdl.dataset[one_n_index][1], "Negative labels must not match"
        
        #UNDO monkey patch
        TinyImageNet.__getitem__ = __original_getitem
    #import torchvision.models as models
    #resnet18 = models.resnet18(pretrained=True)
    
    #resnet18.forward(all_train[0][0].unsqueeze(0)) #the unsqeeze is because resnet only wants batches.
