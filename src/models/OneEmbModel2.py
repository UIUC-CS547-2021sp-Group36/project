import torchvision as tv
import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.models as tvmodels




class OneEmbModel2(torch.nn.Module):
    def __init__(self,resnet= "resnet101",out_features=1000,pretrained=True):
        super(OneEmbModel2, self).__init__()
        self.out_features = out_features

        self.resnet = None

        if resnet == "resnet18":
            self.resnet = tvmodels.resnet18(pretrained=True)
        elif resnet == "resnet50":
            self.resnet = tvmodels.resnet50(pretrained=True)
        elif resnet == "resnet101":
            self.resnet = tvmodels.resnet101(pretrained=True)
        elif resnet == "resnet152":
            self.resnet = tvmodels.resnet152(pretrained=True)
        else:
            raise NotImplemented("I'm sorry, couldn't create inner model {}".format(resnet_name))




        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=8, padding=1, stride=8)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=48, out_channels=500, kernel_size=8, padding=4, stride=4)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=7, padding=3, stride=2)

        self.linearization = torch.nn.Linear(in_features=(1000+500), out_features=self.out_features)



    def forward(self, images):

        rn_embed = self.resnet(images)
        rn_norm = rn_embed.norm(p=2, dim=1, keepdim=True)
        rn_embed = rn_embed.div(rn_norm.expand_as(rn_embed))

        embed = self.conv1(images)
        embed = self.maxpool1(embed)
        embed = self.conv2(embed)
        embed = self.maxpool2(embed)
        embed = embed.reshape(embed.size(0), -1)
        #DEBUG
        print('shape after reshaping: ', embed.shape)
        embed_norm = embed.norm(p=2, dim=1, keepdim=True)
        embed = embed.div(embed_norm.expand_as(embed))

        print('shape after norm: ', embed.shape)

        final_embed = torch.cat([rn_embed, embed], 1)
        #DEBUG
        print('Embed after concatenating: ', final_embed.shape)

        final_embed = self.linearization(final_embed)
        final_norm = final_embed.norm(p=2, dim=1, keepdim=True)
        output = final_embed.div(final_norm.expand_as(final_embed))


        return output





if __name__ == "__main__":



    model = OneEmbModel2()

    #TODO figure out fake batch creation, see pseudocode

    fake_batch = torch.rand(size=(1, 3, 64, 64), dtype=torch.float32)#fake batch of one image

    one_set_of_embeddings = model.forward(fake_batch) #the unsqeeze is because resnet only wants batches.
    print('SHAPE: ', one_set_of_embeddings.shape)
    #check about the properties of one_set_of_embeddings
