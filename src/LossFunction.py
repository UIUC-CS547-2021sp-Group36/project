import torch.nn

#Thanks to Askar for finding this.
#TODO: Make sure to cite the underlying paper in our writeup.

#https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
LossFunction = torch.nn.TripletMarginLoss
