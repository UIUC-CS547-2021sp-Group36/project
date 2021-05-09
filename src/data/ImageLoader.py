#This is just initial experiment code for now.
import random
from collections import OrderedDict
import torchvision

import torch
import torch.utils.data

from typing import Sequence

from .TinyImageNet import TinyImageNet

class ImageFolderSubset(torch.utils.data.dataset.Subset):
    """A class that represents a subset of a torchvision.datasets.ImageFolder
    
    The ImageFolder class has a number of other important properties that aren't reflected in the standard Subset class.
    """
    def __init__(self, dataset: torchvision.datasets.ImageFolder, indices: Sequence[int]) -> None:
        
        sorted_indices = indices.copy()
        sorted_indices.sort() #sorted by label. TODO: don't trust that the labels are pre-sorted
        
        super(ImageFolderSubset, self).__init__(dataset, sorted_indices)
        
        self.__targets = None
        self.__samples = None
        self.__imgs = None
    
    def __dir__(self):
        all_contents = set()
        all_contents.update(dir(super(ImageFolderSubset, self)))
        all_contents.update(dir(self.dataset))
        return list(all_contents)
    
    def __getattr__(self, name):
        return self.dataset.__getattribute__(name)
    
    @property
    def targets(self):
        if self.__targets is None:
            #populate targets list
            self.__targets = torch.empty(len(self.indices),dtype=torch.long,device=torch.device('cpu'))
            for i in range(len(self.indices)):
                self.__targets[i] = self.dataset.targets[self.indices[i]]
        return self.__targets.tolist()
    
    @property
    def samples(self):
        if self.__samples is None:
            self.__samples = [self.dataset.samples[i] for i in self.indices]
        return self.__samples
    
    @property
    def imgs(self):
        if self.__imgs is None:
            self.__imgs = [self.dataset.imgs[i] for i in self.indices]
        return self.__imgs

class ImageFolderLabelIndex(object):
    def __init__(self, dataset:torchvision.datasets.ImageFolder):
        self.dataset = dataset
        
        #super slow and stupid. Could be done while loading.
        #If the classes are already sorted, we could do it with binary search.
        
        #TODO: This only works for ImageFolder, not any general dataset.
        #TODO: This only works for datasets that are ordered by label
        tmp_sizes = torch.IntTensor([self.dataset.targets.count(i) for i in range(len(self.dataset.classes))])
        cs = tmp_sizes.cumsum(0).tolist()
        cs.insert(0,0)
        self.ranges = OrderedDict([(i,(cs[i],cs[i+1])) for i in range(len(cs)-1)])
        self.sizes = OrderedDict([(i,int(j)) for i,j in enumerate(tmp_sizes)])
    
    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.ranges[index]
        elif isinstance(index, str):
            return self.ranges[self.dataset.class_to_idx[index]]
        else:
            raise Exception("Cant query this index like that. : {}".format(index))
    
    def sample_label(self,exclude=None,weighted_classes=True):
        
        
        if exclude is None:
            options = list(self.sizes.keys())
            weights = list(self.sizes.values())
        else:
            options, weights = zip(*[(i,j) for i,j in self.sizes.items() if not i == exclude])
        
        if not weighted_classes:
            weights = None
        
        return random.choices(options,weights)[0]
        
    def sample_item(self, label=None,exclude=None):
        if label is not None and exclude is not None:
            raise Exception("Sampling within and excluding are mutualy exclusive")
        
        if label is not None:
            return random.randrange(*self[label])
        elif exclude is not None:
            return_label = self.sample_label(exclude=exclude,weighted_classes=True)
            return random.randrange(*self[return_label])
        else: #neither label nor exclude
            return random.randrange(0,len(self.dataset))
    
    def label_for_item(self, item_index):
        return self.dataset.targets[item_index]
        

class TripletSamplingDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset:torchvision.datasets.ImageFolder,
                            batch_size=20,
                            shuffle=True,
                            num_workers: int = 0,
                            pin_memory=False,
                            collate_fn=None):
        
        #TODO: Currently using workers causes a memory leak. Fix later.
        #BUG, WORKAROUND
        import warnings
        if num_workers != 0:
            warnings.warn("TripletSamplingDataLoader: num_workers != 0 causes memory leaks and is currently suppressed.")
            num_workers = 0
        #END WORKAROUND
        
        super(TripletSamplingDataLoader, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn if collate_fn is None else collate_fn,
            pin_memory=pin_memory)
        self.label_index = ImageFolderLabelIndex(self.dataset)
    
    def collate_fn(self, somedata):
        #wi = torch.utils.data.get_worker_info()
        
        query_tensor = torch.stack([i[0] for i in somedata])
        labels = [i[1] for i in somedata]
        
        positive_image_indices = [self.label_index.sample_item(label=l) for l in labels]
        negative_image_indices = [self.label_index.sample_item(exclude=l) for l in labels]
        
        positive_image_tensor = torch.stack([self.dataset[i][0] for i in positive_image_indices])
        negative_image_tensor = torch.stack([self.dataset[i][0] for i in negative_image_indices])
        
        if self.pin_memory:
            query_tensor.pin_memory() #consider stacking into a buffer that is already pinned.
            positive_image_tensor.pin_memory()
            negative_image_tensor.pin_memory()
        
        return (query_tensor.detach(), positive_image_tensor.detach(), negative_image_tensor.detach()), torch.IntTensor(labels).detach()
    
    
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

def load_imagefolder(path="/workspace/datasets/tiny-imagenet-200/",split="train",transform=None,is_valid_file=None):
    import torchvision
    from torchvision import datasets, transforms as T
    if transform is None:
        #The training dataset is already sized at 64
        transform = T.Compose([T.Resize(64), T.CenterCrop(64), T.ToTensor()])
    
    def check_valid(path):
        if path.rsplit(".",1)[-1] == "txt":
            #skip .txt files which give bounding boxes.
            return False
        
        try:
            torchvision.datasets.folder.default_loader(path)
            return True
        except:
            print("BAD FILE: {}".format(path))
            return False
        return True
    
    #loaded_dataset = torchvision.datasets.ImageFolder(path,transform=transform,is_valid_file=check_valid)
    loaded_dataset = TinyImageNet(root=path,split=split,transform=transform,is_valid_file=check_valid if is_valid_file is None else is_valid_file)
    return loaded_dataset

def split_imagefolder(dataset:torchvision.datasets.ImageFolder, proportions:Sequence[float]):
    an_index = ImageFolderLabelIndex(dataset)
    
    N_splits = len(proportions)
    indices_for_splits = [list() for i in range(N_splits)]
    
    for l,startstop in an_index.ranges.items():
        some_indices = list(range(*startstop))
        random.shuffle(some_indices)
        
        N_label = startstop[1] - startstop[0]
        
        sizes = [int(p*N_label) for p in proportions]
        #Probably does not sum to N_label.
        #May have some groups getting assigned 0 elements.
        #Make sure to assign unassigned data to those splits first.
        remainder = N_label - sum(sizes)
        for i in range(N_splits):
            if sizes[i] == 0 and remainder > 0:
                sizes[i]+=1
                remainder-=1
        
        #Assign any still unassigned data to a split
        #TODO: Ugly, and surely a closed-form solution exists
        while remainder > 0:
            for i in range(N_splits):
                if remainder > 0:
                    sizes[i]+=1
                    remainder-=1
        
        split_indices_for_class = list()
        for s in sizes:
            consumed = some_indices[:s]
            some_indices = some_indices[s:]
            split_indices_for_class.append(consumed)
        
        assert(len(some_indices) == 0)
        
        for ifs, sifc in zip(indices_for_splits, split_indices_for_class):
            ifs.extend(sifc)
        
    return_splits = list()
    for ifs in indices_for_splits:
        return_splits.append(ImageFolderSubset(dataset, ifs))
    
    return return_splits

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
