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
    class Batch(object):
        """
        This custom batch object makes it possible to know how long the batch is without knowing about its contents
        """
        def __init__(self, X, Y,lenhint=None):
            self.X = X
            self.Y = Y
            self.length = lenhint if lenhint is not None else len(Y)
        
        def __getitem__(self,i):
            if i == 0:
                return self.X
            if i == 1:
                return self.Y
            else:
                raise IndexError("index out of bounds")
        
        def __len__(self):
            return self.length
    
    def __init__(self, dataset:torchvision.datasets.ImageFolder,
                            batch_size=20,
                            shuffle=True,
                            num_workers: int = 0,
                            pin_memory=False,
                            collate_fn=None,
                            sampler=None):
        
        #TODO: Currently using workers causes a memory leak. Fix later.
        #BUG, WORKAROUND
        import warnings
        if num_workers != 0:
            warnings.warn("TripletSamplingDataLoader: num_workers != 0 causes memory leaks and is currently suppressed.")
            num_workers = 0
        #END WORKAROUND
        
        if sampler is not None:
            shuffle=False
        
        super(TripletSamplingDataLoader, self).__init__(dataset,
            batch_size=batch_size,
            sampler=sampler,
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
        
        return self.Batch((query_tensor.detach(), positive_image_tensor.detach(), negative_image_tensor.detach()), torch.IntTensor(labels).detach())
    
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
    print("tests have been moved to the module data.ImageLoader_Tests")
