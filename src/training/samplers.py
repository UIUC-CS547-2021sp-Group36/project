from torch.utils.data.sampler import *

import torch
from torch._six import int_classes as _int_classes
from torch import Tensor

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

T_co = TypeVar('T_co', covariant=True)

class SubepochSampler(Sampler):
    r"""Wraps another sampler to create iterators for sub-epochs.
    
        Unlike the subset random sampler, this guarantees that all data gets seen if the underlying has that guarantee
    
    """
    def __init__(self, sampler: Sampler, subepoch_size: int) -> None:
        self.sampler = sampler
        self.subepoch_size = subepoch_size
        
        self._contained_iterator = iter(self.sampler)
    
    def _inner_next_cycled(self):
        try:
            return next(self._contained_iterator)
        except StopIteration as si:
            self._contained_iterator = iter(self.sampler)
            return next(self._contained_iterator)
        
    def __iter__(self):
        #Should return an iterator with one sub-epoch worth of indices
        retlist = list()
        for i in range(self.subepoch_size):
            retlist.append(self._inner_next_cycled())
        #torch 1.8.0
        #return (retlist, self.sampler.generator)
        #torch 1.5.0
        return iter(retlist)

    def __len__(self):
        return self.subepoch_size


if __name__ == "__main__":
    import torch
    sub_epoch_size = 500
    batch_size = 5
    
    N = 2000
    X = torch.empty(N,5)
    Y = torch.empty(N,1)
    
    ds = torch.utils.data.TensorDataset(X,Y)
    
    rs_sampler = torch.utils.data.RandomSampler(ds)
    
    a_sampler = SubepochSampler(torch.utils.data.RandomSampler(ds),subepoch_size=sub_epoch_size)
    dl = torch.utils.data.DataLoader(ds,batch_size=batch_size,sampler=a_sampler)
    
    
    for i in range(6):
        one_subpoch_of_batches = list(enumerate(dl))
        assert len(one_subpoch_of_batches) == sub_epoch_size/batch_size, "Didn't work."
