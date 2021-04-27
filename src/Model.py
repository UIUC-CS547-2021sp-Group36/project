#This is just initial experiment code for now.
import torchvision as tv
import torch
import torchvision.models as models


#write models here


if __name__ == "__main__":
    from torchvision import datasets, transforms as T
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    #testing

    print("load data")
    all_train = tv.datasets.ImageFolder("/workspace/datasets/tiny-imagenet-200/train",transform=transform)
    
    
    resnet18 = models.resnet18(pretrained=True)
    
    #TODO figure out fake batch creation, see pseudocode
    
    fake_batch = all_train[0][0].unsqueeze(0) #fake batch of one image
    
    one_set_of_embeddings = resnet18.forward(fake_batch) #the unsqeeze is because resnet only wants batches.
    
    #check about the properties of one_set_of_embeddings
