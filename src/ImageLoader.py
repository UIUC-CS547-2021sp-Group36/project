#This is just initial experiment code for now.
import random
from collections import OrderedDict
import torchvision as tv

import torch


class ImageFolderLabelIndex(object):
    def __init__(self, dataset):
        self.dataset = dataset
        
        #super slow and stupid. Could be done while loading.
        #If the classes are already sorted, we could do it with binary search.
        tmp_sizes = torch.IntTensor([self.dataset.targets.count(i) for i in range(len(self.dataset.classes))])
        cs = tmp_sizes.cumsum(0).tolist()
        cs.insert(0,0)
        self.ranges = dict([(i,(cs[i],cs[i+1])) for i in range(len(cs)-1)])
        self.sizes = dict([(i,int(j)) for i,j in enumerate(tmp_sizes)])
    
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

if __name__ == "__main__":
    
    import torch
    from torchvision import datasets, transforms as T
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    #testing
    
    print("load data")
    all_train = tv.datasets.ImageFolder("/workspace/datasets/tiny-imagenet-200/train",transform=transform)
    
    print("index data")
    an_index = ImageFolderLabelIndex(all_train)
    
    print("done indexing data")
    
    print("Example sampling")
    for i in range(20):
        l = an_index.sample_label(weighted_classes=False)
        i_query = an_index.sample_item(label=l)
        i_pos = an_index.sample_item(label=l)
        i_neg = an_index.sample_item(exclude=l)
    
        print(l,i_query, i_pos, i_neg)
    
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    
    resnet18.forward(all_train[0][0].unsqueeze(0)) #the unsqeeze is because resnet only wants batches.
