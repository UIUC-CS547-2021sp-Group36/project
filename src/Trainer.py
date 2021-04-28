import torch
import torch.optim

import ImageLoader
import LossFunction

class Trainer(object):
    def __init__(self, model, dataloader:ImageLoader.TripletSamplingDataLoader, validation_set, g=1.0):
        self.model = model
        self.dataloader = dataloader
        self.validation_set = validation_set
        self.g = g
        self.loss_fn = LossFunction.LossFunction(self.g)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-5) #TODO: not hardcoded
        self.learning_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        
        self.total_epochs = 0
        
    def train(self, n_epochs):
        
        
        for _ in range(n_epochs):
            self.total_epochs += 1
            
            for (Qs,Ps,Ns),l in self.dataloader:
                self.model.train(True)
                self.optimizer.zero_grad()
                
                Q_embedding_vectors = self.model(Qs)
                P_embedding_vectors = self.model(Ps)
                N_embedding_vectors = self.model(Ns)
                
                batch_loss = self.loss_fn(Q_embedding_vectors, P_embedding_vectors, N_embedding_vectors)
                batch_loss.backward()
                
                #TODO: Add proper logging
                #DEBUG
                print("batch loss {} ".format(float(batch_loss)))
                
                self.optimizer.step()
                
                #TODO: Any per-batch logging
                #END of loop over batches
            self.model.train(False)
            
            #TODO: any logging
            #TODO: any validation checking, any learning_schedule stuff.

if __name__ == "__main__":
    import torch
    import torchvision
    
    #testing
    
    print("load data")
    all_train = ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/train")
    
    splits = ImageLoader.split_imagefolder(all_train, [0.02,0.98])
    
    print("create dataloader")
    tsdl = ImageLoader.TripletSamplingDataLoader(splits[0],batch_size=20, num_workers=2)
    
    print("create model")
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    
    print("create trainer")
    test_trainer = Trainer(resnet18, tsdl, None)
    
    print("Begin training")
    test_trainer.train(1)
