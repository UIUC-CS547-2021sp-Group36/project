import torchvision as tv
import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.models as tvmodels

import torch.nn.functional as tnnf

import models.resnet_factory as resnet_factory


class ThreeEmbModel(torch.nn.Module):
    def __init__(self,resnet= "resnet101", out_features=1000, pretrained=True):
        super(ThreeEmbModel, self).__init__()
        self.out_features = out_features

        self.resnet = resnet_factory.create_resnet(resnet,pretrained)

        #self.upsample_rn = torch.nn.Upsample(size=224, mode='bilinear')

        self.downsample1 = torch.nn.Upsample(size=57, mode='bilinear')
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=1, stride=3)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=4)

        self.downsample2 = torch.nn.Upsample(size=29, mode='bilinear')
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=4, stride=6)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=7, padding=3, stride=2)

        self.downsample3 = torch.nn.Upsample(size=15, mode='bilinear')
        self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=4, stride=2)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=7, padding=3, stride=2)

        self.linearization = torch.nn.Linear(in_features=(1000 + 3648), out_features=self.out_features)


    def forward(self, images):

        #images224 = self.upsample_rn(images)
        images224 = images
        rn_embed = self.resnet(images224)
        rn_embed = tnnf.normalize(rn_embed,p=2, dim=1)
        
        
        down_images1 = self.downsample1(images)
        first_embed = self.conv1(down_images1)
        first_embed = self.maxpool1(first_embed)
        first_embed = first_embed.reshape(first_embed.size(0), -1)


        down_images2 = self.downsample2(images)
        second_embed = self.conv2(down_images2)
        second_embed = self.maxpool2(second_embed)
        second_embed = second_embed.reshape(second_embed.size(0), -1)


        down_images3 = self.downsample3(images)
        third_embed = self.conv2(down_images3)
        third_embed = self.maxpool2(third_embed)
        third_embed = third_embed.reshape(third_embed.size(0), -1)



        merge_embed = torch.cat([first_embed, second_embed, third_embed], 1)
        #DEBUG
        #print('Shape after nnorm: ', merge_norm.shape)
        merge_embed = tnnf.normalize(merge_embed,p=2,dim=1)

        #DEBUG
        #print(merge_embed.shape, rn_embed.shape)

        final_embed = torch.cat([rn_embed, merge_embed], 1)
        #DEBUG
        #print(final_embed.shape)
        final_embed = self.linearization(final_embed)
        output = tnnf.normalize(final_embed,p=2,dim=1)

        return output





if __name__ == "__main__":



    model = ThreeEmbModel(resnet="resnet18")

    #TODO figure out fake batch creation, see pseudocode

    fake_batch = torch.rand(size=(2, 3, 64, 64), dtype=torch.float32)#fake batch of one image

    one_set_of_embeddings = model.forward(fake_batch) #the unsqeeze is because resnet only wants batches.
    print('SHAPE: ', one_set_of_embeddings.shape)
    #check about the properties of one_set_of_embeddings
