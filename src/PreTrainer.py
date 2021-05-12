import time

import torch
import torch.optim

import wandb

import data.ImageLoader as ImageLoader
import LossFunction
import Model

class PreTrainingClassifier(torch.nn.Module):
    def __init__(self, embedder, d_dimensions, num_classes=200):
        super(PreTrainingClassifier, self).__init__()
        self.embedder = embedder
        self.num_classes = num_classes
        
        #Doesn't work.
        #embedder_outsize = list(self.embedder.children())[-1].out_features
        
        self.final_layer = torch.nn.Sequential(
                            torch.nn.Linear(d_dimensions,num_classes),
                            torch.nn.ReLU()
        )
    
    def forward(self, data):
        x = self.embedder(data)
        x = self.final_layer(x)
        return x

class PreTrainer(object):
    def __init__(self, model,
            dataloader:ImageLoader.TripletSamplingDataLoader,
            validation_set:ImageLoader.TripletSamplingDataLoader,
            g=1.0,
            num_classes=200):
        self.model = model
        self.dataloader = dataloader
        self.validation_set = validation_set
        self.g = g
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        #FREEZING (search other files.)
        #This should really be done automatically in the optimizer. Not thrilled with this.
        #only optimize parameters that we want to optimize
        optim_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.SGD(optim_params, lr=0.01, weight_decay=1e-5) #TODO: not hardcoded
        self.learning_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        
        self.total_epochs = 0
        
        #Logging
        self.batch_log_interval = 10
        
    def train(self, n_epochs):
        
        
        for _ in range(n_epochs):
            self.total_epochs += 1
            
            for batch_idx, (img,l) in enumerate(self.dataloader):
                batch_start = time.time() #Throughput measurement
                
                self.model.train(True)
                self.optimizer.zero_grad()
                
                predicted_classes = self.model(img)
                
                batch_loss = self.loss_fn(predicted_classes, l)
                batch_loss.backward()
                
                self.optimizer.step()
                
                batch_end = time.time() #Throughput measurement
                batch_time_per_item = float(batch_end-batch_start)/len(l) #Throughput measurement
                
                #TODO: Add proper logging
                #DEBUG
                print("batch loss {} time {}s/item".format(float(batch_loss), batch_time_per_item))
                wandb.log({"batch_loss":float(batch_loss), "time_per_item":batch_time_per_item})
                
                #TODO: Any per-batch logging
                #END of loop over batches
            self.model.train(False)
            
            #TODO: any logging
            #TODO: any validation checking, any learning_schedule stuff.

if __name__ == "__main__":
    import torch
    import torchvision
    
    #testing
    wandb.init(
                entity='uiuc-cs547-2021sp-group36',
                project='image_similarity',
                group="debugging")
    
    print("load data")
    all_train = ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/train")
    
    train_data, crossval_data = ImageLoader.split_imagefolder(all_train, [0.95,0.05])
    
    print("create dataloader")
    tsdl = torch.utils.data.DataLoader(train_data,batch_size=200, shuffle=True, num_workers=1)
    #tsdl = ImageLoader.TripletSamplingDataLoader(train_data,batch_size=200, num_workers=1)
    
    print("create model")
    model = Model.create_model("dummy")
    
    wandb.watch(model, log_freq=100)
    
    print("create trainer")
    pmodel = PreTrainingClassifier(model,300,200)
    test_trainer = PreTrainer(pmodel, tsdl, None)
    
    print("Begin training")
    test_trainer.train(100)
