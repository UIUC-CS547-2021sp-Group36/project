#Much of this code is copied/adapted from torchvision.datasets.folder.ImageNet

import warnings
from contextlib import contextmanager
import os
import shutil
import tempfile
from typing import Any, Dict, List, Iterator, Optional, Tuple
import torch

from torchvision.datasets.folder import ImageFolder

class TinyImageNet(ImageFolder):
    """`TinyImageNet <http://image-net.org/>`_ 2012 Classification Dataset.
    
    Copied from torchvision sorce and updated.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(self, root: str, split: str = 'train', download: Optional[str] = None, **kwargs: Any) -> None:
        if download is True:
            #TODO: TinyImageNet is actually available.
            msg = ("The dataset is no longer publicly accessible. You need to "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)
        elif download is False:
            msg = ("The use of the download flag is deprecated, since the dataset "
                   "is no longer publicly accessible.")
            warnings.warn(msg, RuntimeWarning)
        
        root = self.root = os.path.expanduser(root)
        self.split = split
        if self.split not in ["train", "val"]:
            raise Exception("The split {} is not available. Sorry.".format(split))
        
        self.split_folder = None
        if self.split == "train":
            self.split_folder = os.path.join(self.root,"train")
        elif self.split == "val":
            self.split_folder = os.path.join(self.root,"val")
        
        super(TinyImageNet, self).__init__(self.split_folder, **kwargs)
        
        self.root = root #to override the root from ImageFolder, keep root as overall root
        
        wnids = self.load_wnid_file(os.path.join(self.root,"wnids.txt"))
        wnids_to_id = {cls: idx for idx, cls in enumerate(wnids)}
        
        if self.split == "train":
            #self.classes may not have loaded in the order of the wnids.txt file, or may be redacted
            
            loaded_class_id_to_wnid = {self.class_to_idx[cls]: wnids_to_id[cls] for cls in self.classes}
            
            tmpsamples = [(f,loaded_class_id_to_wnid[cls]) for f,cls in self.samples]
            self.samples.clear()#inplace
            self.samples.extend(tmpsamples)#inplace
            del tmpsamples
            self.samples.sort(key=lambda x:x[1])
            #self.imgs = self.samples #Because of ImageFolder
            
            self._loaded_classes = self.classes #TODO: comment after debug
            self.classes = wnids
            
            self._loaded_class_to_idx = self.class_to_idx #TODO: Comment after debug
            self.class_to_idx = wnids_to_id
        elif self.split == "val":
            val_classes,_ = self.load_crossval_targets(os.path.join(self.root,"val","val_annotations.txt"))
            tmpsamples = [(f,wnids_to_id[val_classes[os.path.basename(f)]]) for f,c in self.samples]
            self.samples.clear()#inplace
            self.samples.extend(tmpsamples)#inplace
            del tmpsamples
            self.samples.sort(key=lambda x:x[1])
            #self.imgs = self.samples
            
            self.classes = wnids
            self.class_to_idx = wnids_to_id
        else:
            raise Exception("Invalid split {}".format(self.split))
        #From ImageFolder
        self.targets = [s[1] for s in self.samples] #I wish this were a property
            
    @classmethod
    def load_wnid_file(cls, path):
        lines = None
        with open(path,"r") as infile:
            lines = [l.strip() for l in infile]
        return lines
    
    @classmethod
    def load_crossval_targets(cls, path):
        lines = None
        with open(path,"r") as infile:
            lines = [l.strip().split() for l in infile]
        class_mapping = {l[0]: l[1] for l in lines}
        bb_mapping = {l[0]: l[2:] for l in lines}
        
        return class_mapping, bb_mapping

if __name__ == "__main__":
    #import data.ImageLoader as ImageLoader
    print("Running tests of TinyImageNet class")
    
    a = ImageFolder("/workspace/datasets/tiny-imagenet-200/train")
    
    tin_train = TinyImageNet("/workspace/datasets/tiny-imagenet-200/","train",is_valid_file=lambda x: x.rsplit(".",1)[-1] == "JPEG")
    for ffn, cls in tin_train.samples:
        _, fn = os.path.split(ffn)
        cls_from_fn = fn.split("_",1)[0]
        assert tin_train.classes.index(cls_from_fn) == tin_train.class_to_idx[cls_from_fn] == cls, "Classes do not map"
    print("passed on train split")
    
    print("testing val split")
    #tin_val = TinyImageNet("/workspace/datasets/tiny-imagenet-200/","val",is_valid_file=lambda x: x.rsplit(".",1)[-1] == "JPEG")
    tin_val = TinyImageNet("/workspace/datasets/tiny-imagenet-200/","val")
    assert tin_train.classes == tin_val.classes, "Classes don't match"
    assert tin_train.class_to_idx == tin_val.class_to_idx, "Mappings don't match"
    
