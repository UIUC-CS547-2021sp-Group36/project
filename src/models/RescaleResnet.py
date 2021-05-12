#This is just initial experiment code for now.
import torchvision as tv
import torch
import torchvision.models as models
import torchvision.models as tvmodels
import torchvision.transforms


import torch.nn.functional as tnnf

import models.resnet_factory as resnet_factory

torchvision_versions = list(map(int,tv.__version__.split(".")[:2]))

import warnings

#write models here
class RescaleResnet(torch.nn.Module):
    
    class RescaleFallback(torch.nn.Module):
        """
        Module to wrap torch.nn.functional.interpolate
        """
        def __init__(self, size=None,
                            scale_factor=None,
                            mode='nearest',
                            align_corners=None,
                            recompute_scale_factor=None):
            super(RescaleResnet.RescaleFallback, self).__init__()
            self.size = size
            self.scale_factor = scale_factor
            self.mode = mode
            self.align_corners = align_corners
            self.recompute_scale_factor = recompute_scale_factor
        
        def forward(self,input):
            return torch.nn.functional.interpolate(input,
                                    size=self.size,
                                    scale_factor=self.scale_factor,
                                    mode=self.mode,
                                    align_corners=self.align_corners,
                                    recompute_scale_factor=self.recompute_scale_factor
                                    )
            
            
    
    def __init__(self, resnet="resnet18", scale=(224,224), freeze_resnet=True,pretrained=True):
        super(RescaleResnet, self).__init__()
        self.resnet = resnet_factory.create_resnet(resnet,pretrained)
        self.scale = scale
        self.transform = None
        
        if torchvision_versions[0] == 0 and torchvision_versions[1] <= 6:
            #Can't call scale on a Tensor in old torchvision
            warnings.warn("Old version of torchvision, using fallback rescale.")
            self.transform = torch.nn.Upsample(size=scale, mode='bilinear')
        else:
            self.transform = torchvision.transforms.Resize(self.scale)
        
        #FREEZING (search other files)
        #This is supposed to help freeze the submodel, but the optimizer
        #does not respect this alone. It's very sad.
        if freeze_resnet:
            self.resnet.requires_grad = False

    def forward(self, images):
        images = self.transform(images)
        rn_embed = self.resnet(images)

        #Normalize output while training
        #Nevermind, we'll use a loss function to enforce that
        #if self.training:
        #    output = torch.nn.functional.normalize(output,dim=1)

        return rn_embed
