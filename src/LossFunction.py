import torch
import torch.nn

import torch.nn.functional as _F

#Thanks to Askar for finding this.
#TODO: Make sure to cite the underlying paper in our writeup.

#https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
LossFunction = torch.nn.TripletMarginLoss


class SphereNormLoss(torch.nn.modules.loss._Loss):
    def __init__(self, r: float = 1.0, r2: float = 2.0, r_strength: float = 1.0,
                    size_average=None,
                    reduce=None, reduction: str = 'mean'):
        super(SphereNormLoss, self).__init__(size_average, reduce, reduction)
        self.r = r #must be positive
        self.r2 = r2 #must be positive and greater than r
        self.r_strength = r_strength #negative to keep in, positive to keep out
    
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        #distance_inside will be negative if the vectors are outside the r-sphere
        n = torch.norm(vectors,dim=1)
        d_in = self.r - n
        d_in = d_in * self.r_strength #Doing this first automatically handles which side it should be on.
        
        d_out = n - self.r2
        d_out = d_out * self.r_strength
        x = torch.nn.functional.relu(d_in) + torch.nn.functional.relu(d_out)
        
        
        if "mean" == self.reduction:
            x = x.mean()
        else:
            raise NotImplemented("Can't do reduction {}".format(self.reduction))
        
        return x

class SphereTML(torch.nn.modules.loss._Loss):
    def __init__(self, margin: float = 1.0, r: float = 1.0, r_strength: float = 1.0,
                    p: float = 2., eps: float = 1e-6,
                    swap: bool = False, size_average=None,
                    reduce=None, reduction: str = 'mean'):
        super(SphereTML, self).__init__(size_average, reduce, reduction)
        self.margin = margin
        self.r = r
        self.tml = torch.nn.TripletMarginLoss(margin = self.margin, p=p,eps=eps,swap=swap,size_average=size_average,reduce=reduce,reduction=reduction)
        self.snl = SphereNormLoss(r,r_strength, size_average = size_average, reduce = reduce, reduction=reduction)
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        tml_loss = self.tml(anchor, positive, negative)
        a_norm_loss = self.snl(anchor)
        p_norm_loss = self.snl(positive)
        n_norm_loss = self.snl(negative)
        
        return tml_loss + a_norm_loss + p_norm_loss + n_norm_loss


class NormedTML(torch.nn.TripletMarginLoss):
    def __init__(self, *args,**kwargs):
        super(NormedTML, self).__init__(*args,**kwargs)
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        
        a = torch.nn.functional.normalize(anchor)
        p = torch.nn.functional.normalize(positive)
        n = torch.nn.functional.normalize(negative)
        
        return super(NormedTML, self).forward(a,p,n)
        
class TripletAccuracy(torch.nn.Module):
    def __init__(self, *args,**kwargs):
        super(TripletAccuracy,self).__init__()
        self.pairwise = torch.nn.PairwiseDistance(p=2.0)
        self.reduction = "sum"
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        
        dist_q_p = self.pairwise(anchor, positive)
        dist_q_n = self.pairwise(anchor, negative)
        differences = torch.lt(dist_q_p, dist_q_n)
        
        #TODO: add an option to use sigmoid and be differentiable.
        return differences.sum()

def create_loss(name="default"):
    #TODO: Handle additional arguments and pass them to the constructor
    if name in [None, "default","TripletMarginLoss","torch.nn.TripletMarginLoss"]:
        return torch.nn.TripletMarginLoss(margin=1.0)
    elif name in ["sphere","sphere_tml"]:
        return SphereTML(margin=1.0)
    elif name in ["normed"]:
        return NormedTML(margin=1.0)
    elif name in ["cosine","triplet_cosine"]:
        #Not available in 1.5.0!
        return torch.nn.TripletMarginWithDistanceLoss(margin=1.0,distance_function=torch.nn.CosineSimilarity())
    
    #TODO: Add options for other models as we implement them.
    
    raise Exception("No or invalid loss requested : '{}' ".format(name))
