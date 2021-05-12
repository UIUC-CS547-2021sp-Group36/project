import torchvision as tv
import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.models as tvmodels


import torch.nn.functional as tnnf

import models.resnet_factory as resnet_factory


class OneEmbModel2(torch.nn.Module):
    def __init__(self,resnet= "resnet101",out_features=1000,pretrained=True):
        super(OneEmbModel2, self).__init__()
        self.out_features = out_features

        self.resnet = resnet_factory.create_resnet(resnet,pretrained)



        #self.upsample_rn = torch.nn.Upsample(size=224, mode='bilinear')

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=8, padding=1, stride=8)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=48, out_channels=500, kernel_size=8, padding=4, stride=4)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=7, padding=3, stride=2)

        self.linearization = torch.nn.Linear(in_features=(1000+500), out_features=self.out_features)



    def forward(self, images):

        #images224 = self.upsample_rn(images)
        images224 = images
        rn_embed = self.resnet(images224)
        rn_embed = tnnf.normalize(rn_embed, p=2, dim=1)

        embed = self.conv1(images)
        embed = self.maxpool1(embed)
        embed = self.conv2(embed)
        embed = self.maxpool2(embed)
        embed = embed.reshape(embed.size(0), -1)
        #DEBUG
        # print('shape after reshaping: ', embed.shape)
        embed = tnnf.normalize(embed, p=2, dim=1)

        # print('shape after norm: ', embed.shape)

        final_embed = torch.cat([rn_embed, embed], 1)
        #DEBUG
        # print('Embed after concatenating: ', final_embed.shape)

        final_embed = self.linearization(final_embed)
        output = tnnf.normalize(final_embed, p=2, dim=1)

        return output





if __name__ == "__main__":



    model = OneEmbModel2()

    #TODO figure out fake batch creation, see pseudocode

    fake_batch = torch.rand(size=(1, 3, 64, 64), dtype=torch.float32)#fake batch of one image

    one_set_of_embeddings = model.forward(fake_batch) #the unsqeeze is because resnet only wants batches.
    print('SHAPE: ', one_set_of_embeddings.shape)
    #check about the properties of one_set_of_embeddings
