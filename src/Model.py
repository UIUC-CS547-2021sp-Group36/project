#This is just initial experiment code for now.
import torchvision as tv
import torch
import torchvision.models as models


#write models here



def create_model(model_name="resnet18"):
    if model_name is "resnet18":
        return models.resnet18(pretrained=True)
    
    #TODO: Add options for other models as we implement them.
    
    raise Exception("No Model Specified")

if __name__ == "__main__":
    import ImageLoader
    

    print("load data")
    all_train = ImageLoader.load_imagefolder()
    train, val = ImageLoader.split_imagefolder(all_train, [0.9,0.1])
    
    
    resnet18 = models.resnet18(pretrained=True)
    
    #TODO figure out fake batch creation, see pseudocode
    
    fake_batch = all_train[0][0].unsqueeze(0) #fake batch of one image
    
    one_set_of_embeddings = resnet18.forward(fake_batch) #the unsqeeze is because resnet only wants batches.
    
    #check about the properties of one_set_of_embeddings
