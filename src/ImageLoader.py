#This is just initial experiment code for now.
import random
from collections import OrderedDict
import torchvision as tv

import torch
import torch.utils.data


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
    def __init__(self, dataset,
                            batch_size=20,
                            shuffle=True,
                            num_workers: int = 0,
                            pin_memory=False):
        super(TripletSamplingDataLoader, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
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
        
    tsdl = TripletSamplingDataLoader(all_train,batch_size=20, num_workers=2)
    
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
        
    
    
    #import torchvision.models as models
    #resnet18 = models.resnet18(pretrained=True)
    
    #resnet18.forward(all_train[0][0].unsqueeze(0)) #the unsqeeze is because resnet only wants batches.
