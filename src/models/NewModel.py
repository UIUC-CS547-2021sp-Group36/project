#This is just initial experiment code for now.
import torchvision as tv
import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.models as tvmodels




class NewModel(torch.nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()

        self.resnet = tvmodels.resnet18(pretrained=True)
        self.resnet.requires_grad = False

        self.downsample1 = torch.nn.Upsample(size=57, mode='bilinear')
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=1, stride=3)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=4)

        self.downsample2 = torch.nn.Upsample(size=29, mode='bilinear')
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=4, stride=6)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=7, padding=3, stride=2)

        self.linearization = torch.nn.Linear(in_features=(1000 + 3264), out_features=1000)


    def forward(self, images):

        rn_embed = self.resnet(images)
        norm = rn_embed.norm(p=2, dim=1, keepdim=True)
        second_input = rn_embed.div(norm.expand_as(rn_embed))

        down_images1 = self.downsample1(images)
        first_embed = self.conv1(down_images1)
        first_embed = self.maxpool1(first_embed)
        first_embed = first_embed.reshape(first_embed.size(0), -1)


        down_images2 = self.downsample2(images)
        second_embed = self.conv2(down_images2)
        second_embed = self.maxpool2(second_embed)
        second_embed = second_embed.reshape(second_embed.size(0), -1)



        merge_embed = torch.cat([first_embed, second_embed], 1)
        merge_norm = merge_embed.norm(p=2, dim=1, keepdim=True)
        #DEBUG
        #print('Shape after nnorm: ', merge_norm.shape)
        merge_embed = merge_embed.div(merge_norm.expand_as(merge_embed))

        #DEBUG
        #print(merge_embed.shape, rn_embed.shape)

        final_embed = torch.cat([rn_embed, merge_embed], 1)
        #DEBUG
        #print(final_embed.shape)
        final_embed = self.linearization(final_embed)
        final_norm = final_embed.norm(p=2, dim=1, keepdim=True)
        output = final_embed.div(final_norm.expand_as(final_embed))


        return output





if __name__ == "__main__":



    model = NewModel()

    #TODO figure out fake batch creation, see pseudocode

    fake_batch = torch.rand(size=(1, 3, 64, 64), dtype=torch.float32)#fake batch of one image

    one_set_of_embeddings = model.forward(fake_batch) #the unsqeeze is because resnet only wants batches.
    print('SHAPE: ', one_set_of_embeddings.shape)
    #check about the properties of one_set_of_embeddings
