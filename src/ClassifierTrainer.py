import time
import os, signal

import torch
import torch.nn
import torch.optim

import wandb

import ImageLoader
import LossFunction
import models

class ClassifierAccuracy(torch.nn.Module):
    def __init__(self,reduction:str="sum"):
        super(ClassifierAccuracy,self).__init__()
        self.reduction = reduction
    
    def forward(self, predictions, labels):
        num_correct = (predictions.argmax(1) == labels)
        if reduction in ["sum","mean"]:
            num_correct = num_correct.sum()
        if reduction in ["mean"]:
            num_correct /= float(labels.size()[0])
        return num_correct

class ClassifierWrapper(torch.nn.Module):
    def __init__(self,other,n_classes):
        super(ClassifierWrapper,self).__init__()
        self.other = other
        self.out_features = n_classes
        
        other_outfeatures = self.find_outfeatures(self.other)
        
        self.final_layers = torch.nn.Sequential(
                    torch.nn.Linear(in_features=other_outfeatures, out_features=200),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(in_features=200,out_features=n_classes),
                    torch.nn.Sigmoid()
                    )
    
    def forward(self, x):
        y = self.other(x)
        y = self.final_layers(y)
        return y
    
    @classmethod
    def find_outfeatures(cls, thing:torch.nn.Module):
        of = None
        if hasattr(thing,"out_features"):
            return thing.out_features
        elif hasattr(thing,"children"):
            for one_child in thing.children():
                candidate_of = cls.find_outfeatures(one_child)
                if candidate_of is not None:
                    of = candidate_of
        return of

class ClassifierTrainer(object):
    def __init__(self, model,
            dataloader,
            validation_set,
            g=1.0,
            verbose=True,
            lr=0.0001,
            weight_decay=0.00001,
            n_classes=200):
        self.inner_model = model
        self.model = ClassifierWrapper(self.inner_model,n_classes)
        self.dataloader = dataloader
        self.validation_set = validation_set
        self.g = g
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy_function = ClassifierAccuracy()
        
        #FREEZING (search other files.)
        #This should really be done automatically in the optimizer. Not thrilled with this.
        #only optimize parameters that we want to optimize
        optim_params = [p for p in self.model.parameters() if p.requires_grad]
        
        #Optimization
        self.optimizer = torch.optim.SGD(optim_params, lr=lr, weight_decay=weight_decay) #TODO: not hardcoded
        self.lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        
        self.total_epochs = 0
        
        #Various intervals
        self.lr_interval = 10
        
        #Logging
        self.verbose = verbose #log to terminal too?
        self.batch_log_interval = 1
        self.checkpoint_interval = 50 #additional checkpoints in number of batches
        self.monitor_norms = False
        
        #For clean interruption
        self.running = False
        self.previous_handler = None
    
    def handle_sigint(self, signum, frame):
        self.running = False
    
    def create_checkpoint(self):
        if self.verbose:
            print("Creating checkpoint")
        model_file = os.path.join(wandb.run.dir, "model_state.pt")
        trainer_file = os.path.join(wandb.run.dir,"trainer.pt")
        
        torch.save(self.inner_model.state_dict(), model_file)
        wandb.save(model_file)
        
        wrapper_file = os.path.join(wandb.run.dir, "wrapper_state.pt")
        torch.save(self.model.state_dict(), wrapper_file)
        wandb.save(wrapper_file)
        
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
    def crossval(self):
        self.model.eval()
        total_validation_loss = 0.0
        total_seen = 0
        for batch_idx, (Qs,l) in enumerate(self.validation_set):
            y_hat = self.model(Qs).detach()
            
            total_validation_loss += float(self.accuracy_function(y_hat, l))
            total_seen += int(len(l))
        
        total_validation_loss /= float(total_seen)
        total_validation_loss = 1.0 - total_validation_loss
        
        if self.verbose:
            print("Crossval_error {}".format(total_validation_loss))
        wandb.log({"epoch_val_error":total_validation_loss},step=wandb.run.step)
        
        return total_validation_loss
    
    def train(self, n_epochs):
        self.running = True
        #install signal handler for clean interruption
        self.previous_handler = signal.signal(signal.SIGINT, self.handle_sigint)
        
        for _ in range(n_epochs):
            if not self.running:
                break
            self.total_epochs += 1
            
            epoch_average_batch_loss = 0.0;
            batchgroup_average_batch_loss = 0.0;
            
            for batch_idx, (Qs,l) in enumerate(self.dataloader):
                
                if not self.running:
                    break
                
                batch_start_time = time.time() #Throughput measurement
                
                self.model.train(True)
                self.optimizer.zero_grad()
                
                y_hat = self.model(Qs)
                
                batch_loss = self.loss_fn(y_hat, l)
                batch_loss.backward()
                
                self.optimizer.step()
                
                batch_loss = float(batch_loss)
                
                #cumulative
                epoch_average_batch_loss += float(batch_loss)
                batchgroup_average_batch_loss += float(batch_loss)
                
                #creates a step
                wandb.log({"batch_loss":float(batch_loss)},commit=False)
                #DEBUG LOG loss to terminal #TODO: Can wandb echo to terminal?
                if self.verbose and 0 == batch_idx % self.batch_log_interval:
                    print("batch ({}) loss {:.5f}".format(batch_idx,
                                                            float(batch_loss))
                        )
                #TODO: Add proper logging
                
                ## DEBUG: Monitor Norms
                if self.monitor_norms:
                    overall_mean_norms = norm_logging(Q_embedding_vectors, P_embedding_vectors, N_embedding_vectors)
                
                #CHECKPOINTING (epochs are so long that we need to checkpoitn more freqently)
                if 0 != batch_idx and 0 == batch_idx%self.checkpoint_interval:
                    self.create_checkpoint()
                
                #LEARNING SCHEDULE
                if 0 != batch_idx and 0 == batch_idx%self.lr_interval:
                    batchgroup_average_batch_loss /= self.lr_interval
                    self.lr_schedule.step(batchgroup_average_batch_loss)
                    batchgroup_average_batch_loss = 0.0
                    
                    
                    #Any logging of LR rate
                
                #TODO: Any per-batch logging
                #END of loop over batches
                batch_end_time = time.time() #Throughput measurement
                batch_time_per_item = float(batch_end_time-batch_start_time)/len(l) #Throughput measurement
                #Commit wandb logs for this batch
                wandb.log({"time_per_item":batch_time_per_item},commit=True, step=wandb.run.step)
                
                
            self.model.train(False)
            
            #Until now, this was actually total batch loss
            epoch_average_batch_loss /= batch_idx
            wandb.log({"epoch_average_batch_loss":epoch_average_batch_loss
                        },step=wandb.run.step)
            #TODO: log LR for this epoch.
            
            #TODO: any logging
            #TODO: any validation checking, any learning_schedule stuff.
            if self.running and 0 == self.total_epochs % self.lr_interval:
                self.model.eval()
                
                #TODO: blah. Too slow.
                #self.lr_schedule.step(epoch_average_batch_loss)
            
            #CROSSVALIDATION
            if self.running and None != self.validation_set:
                self.crossval()
                        
                        
            
            #Also save a checkpoint after every epoch and upon cancellation
            self.create_checkpoint()
        
        #END of train
        #restore the previous interrupt handler
        signal.signal(signal.SIGINT, self.previous_handler)
        
        return self.running #should_continue

if __name__ == "__main__":
    import os
    import torch
    import torchvision
    
    #testing
    run_id = wandb.util.generate_id()
    #TODO: Move to a main script and a bash script outside this program.
    wandb_tags = ["debug"]
    wandb.init(id=run_id,
                resume="allow",
                entity='uiuc-cs547-2021sp-group36',
                project='image_similarity',
                group="debugging",
                tags=wandb_tags)
    if wandb.run.resumed:
        print("Resuming...")
    
    print("create model")
    model = models.create_model("LowDNewModel")
    if wandb.run.resumed:
        print("Resuming from checkpoint")
        model_pickle_file = wandb.restore("model_state.pt")
        model.load_state_dict( torch.load(model_pickle_file.name) )
    wandb.watch(model, log_freq=100) #Won't work if we restore the full object from pickle. :(
    
    
    print("load data")
    all_train = ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/")
    train_data, crossval_data, _ = ImageLoader.split_imagefolder(all_train, [0.3,0.1,0.1])
    print("create dataloader")
    tsdl = torch.utils.data.DataLoader(train_data,batch_size=100, num_workers=0)
    tsdl_crossval = torch.utils.data.DataLoader(crossval_data,batch_size=100, num_workers=0,shuffle=False)
    
    print("create trainer")
    test_trainer = ClassifierTrainer(model, tsdl, tsdl_crossval,n_classes=200)
    
    print("Begin training")
    test_trainer.train(100)
