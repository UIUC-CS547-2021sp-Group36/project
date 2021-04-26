#This is just initial experiment code for now.
import random
from collections import OrderedDict
import torchvision as tv


class ImageFolderLabelIndex(object):
    def __init__(self, dataset):
        self.dataset = dataset
        
        #super slow and stupid. Could be done while loading.
        #If the classes are already sorted, we could do it with binary search.
        current_class = None
        class_firstindex = list()
        class_list = list() #don't assume they are sorted.
        for i, item in enumerate(self.dataset):
            if item[1] != current_class:
                current_class = item[1]
                class_firstindex.append(i)
                class_list.append(current_class)

        class_firstindex.append(len(self.dataset))
        
        class_ranges = list()
        
        assert len(class_list) + 1 == len(class_firstindex)
        
        for i, c in enumerate(class_list):
            class_ranges.append((c,(class_firstindex[i], class_firstindex[i+1])))
        self.ranges = dict(class_ranges)
        self.sizes = dict([(label,end-begin) for label,(begin,end) in class_ranges])
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.ranges[index]
        elif isinstance(index, str):
            return self.ranges[self.dataset.class_to_idx[index]]
        else:
            raise Exception("Cant query this index like that.")
    
    def sample_label(self,exclude=None,weighted_classes=True):
        
        
        if exclude is None:
            options = self.sizes.keys()
            weights = self.sizes.values()
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
            return_label = self.sample_label(label,exclude=label)
            return random.randrange(*self[label])
        else: #neither label nor exclude
            return random.randrange(0,len(self.dataset))

if __name__ == "__main__":
    #testing
    
    print("load data")
    all_train = tv.datasets.ImageFolder("/workspace/datasets/tiny-imagenet-200/train")
    
    print("index data")
    an_index = ImageFolderLabelIndex(all_train)
    
    print("done indexing data")
