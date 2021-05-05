#This is just initial experiment code for now.
import torchvision as tv
import torch
import torchvision.models as models
import torchvision.models as tvmodels


#write models here
class Resnet18FrozenWrapper(torch.nn.Module):
    def __init__(self, output_dimension=300,internal_dimension=500):
        super(Resnet18FrozenWrapper, self).__init__()
        self.resnet = tvmodels.resnet18(pretrained=True)

        #We find the dimensions of the output from resnet
        #that will be the dimension of the input for the subesquent layer
        n_resnetout = list(self.resnet.children())[-1].out_features

        self.additional_layers = torch.nn.Sequential(
            torch.nn.Linear(n_resnetout, internal_dimension),
            torch.nn.Sigmoid(),
            torch.nn.Linear(internal_dimension,output_dimension),
            torch.nn.Sigmoid()
            #torch.nn.Softmax(dim=1) #More classifier like
        )

        #FREEZING (search other files)
        #This is supposed to help freeze the submodel, but the optimizer
        #does not respect this alone. It's very sad.
        self.resnet.requires_grad = False

    def forward(self, images):
        rn_embed = self.resnet(images)
        output = self.additional_layers(rn_embed)

        #Normalize output while training
        #Nevermind, we'll use a loss function to enforce that
        #if self.training:
        #    output = torch.nn.functional.normalize(output,dim=1)

        return output


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
