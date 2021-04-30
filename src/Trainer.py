import time
import os

import torch
import torch.optim

import wandb

import ImageLoader
import LossFunction
import Model

class Trainer(object):
    def __init__(self, model,
            dataloader:ImageLoader.TripletSamplingDataLoader,
            validation_set:ImageLoader.TripletSamplingDataLoader,
            g=1.0,
            verbose=True):
        self.model = model
        self.dataloader = dataloader
        self.validation_set = validation_set
        self.g = g
        self.loss_fn = LossFunction.LossFunction(self.g)
        
        #FREEZING (search other files.)
        #This should really be done automatically in the optimizer. Not thrilled with this.
        #only optimize parameters that we want to optimize
        optim_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.Adam(optim_params, lr=0.1) #TODO: not hardcoded
        self.learning_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        
        self.total_epochs = 0
        
        #Various intervals
        self.lr_interval = 10
        
        #Logging
        self.verbose = verbose #log to terminal too?
        self.batch_log_interval = 5
        self.checkpoint_interval = 1 #epochs
    
    def create_checkpoint(self):
        if self.verbose:
            print("Creating checkpoint")
        model_file = os.path.join(wandb.run.dir, "model_state.pt")
        trainer_file = os.path.join(wandb.run.dir,"trainer.pt")
        
        torch.save(self.model.state_dict(), model_file)
        wandb.save(model_file)
        
        #This will save the entire dataset and everything.
        #We need to implement a state_dict kind of thing for the trainer.
        #torch.save(self, trainer_file)
        #wandb.save(trainer_file)
    
    _ = """
    @classmethod
    def load_checkpoint(cls,base_dir=None):
        raise Exception("Currently unavailable")
        if None == base_dir:
            base_dir = wandb.run.dir
        
        trainer_file = os.path.join(base_dir,"trainer.pt")
        unpickled_trainer = torch.load(model_file)
        return unpickled_trainer
    """
        
    def train(self, n_epochs):
        
        
        for _ in range(n_epochs):
            self.total_epochs += 1
            
            epoch_average_batch_loss = 0.0;
            
            for batch_idx, ((Qs,Ps,Ns),l) in enumerate(self.dataloader):
                batch_start = time.time() #Throughput measurement
                
                self.model.train(True)
                self.optimizer.zero_grad()
                
                Q_embedding_vectors = self.model(Qs)
                P_embedding_vectors = self.model(Ps)
                N_embedding_vectors = self.model(Ns)
                
                batch_loss = self.loss_fn(Q_embedding_vectors, P_embedding_vectors, N_embedding_vectors)
                batch_loss.backward()
                
                self.optimizer.step()
                
                batch_end = time.time() #Throughput measurement
                batch_time_per_item = float(batch_end-batch_start)/len(l) #Throughput measurement
                
                epoch_average_batch_loss += float(batch_loss)
                #TODO: Add proper logging
                #DEBUG
                if self.verbose and 0 == batch_idx % self.batch_log_interval:
                    print("batch ({}) loss {} time {}s/item".format(batch_idx,
                                                            float(batch_loss),
                                                            batch_time_per_item)
                        )
                wandb.log({"batch_loss":float(batch_loss),
                            "time_per_item":batch_time_per_item
                            })
                
                #TODO: Any per-batch logging
                #END of loop over batches
            self.model.train(False)
            
            #Until now, this was actually total batch loss
            epoch_average_batch_loss /= batch_idx
            wandb.log({"epoch_average_batch_loss":epoch_average_batch_loss
                        })
            
            #TODO: any logging
            #TODO: any validation checking, any learning_schedule stuff.
            if 0 == self.total_epochs % self.lr_interval:
                self.model.eval()
                
                #TODO: blah. Too slow.
                self.lr_scheduler.step(epoch_average_batch_loss)
            
            self.create_checkpoint()
                

if __name__ == "__main__":
    import torch
    import torchvision
    
    #testing
    run_id = wandb.util.generate_id()
    wandb.init(id=run_id,
                resume="allow",
                entity='uiuc-cs547-2021sp-group36',
                project='image_similarity',
                group="debugging",
                tags=["debug"])
    if wandb.run.resumed:
        print("Resuming...")
    
    print("create model")
    model = Model.create_model("dummy")
    if wandb.run.resumed:
        print("Resuming from checkpoint")
        model_pickle_file = wandb.restore("model_state.pt")
        model.load_state_dict( torch.load(model_pickle_file.name) )
    wandb.watch(model, log_freq=100) #Won't work if we restore the full object from pickle. :(
    
    
    print("load data")
    all_train = ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/train")
    train_data, crossval_data = ImageLoader.split_imagefolder(all_train, [0.2,0.7])
    print("create dataloader")
    tsdl = ImageLoader.TripletSamplingDataLoader(train_data,batch_size=200, num_workers=0)
    
    print("create trainer")
    test_trainer = Trainer(model, tsdl, None)
    
    print("Begin training")
    test_trainer.train(10)
