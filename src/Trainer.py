import time
import os

import torch
import torch.nn
import torch.optim

import wandb

import ImageLoader
import LossFunction
import models

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
        self.accuracy_function = torch.nn.TripletMarginLoss(margin = 0.0,reduction="sum")
        
        #FREEZING (search other files.)
        #This should really be done automatically in the optimizer. Not thrilled with this.
        #only optimize parameters that we want to optimize
        optim_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.Adam(optim_params, lr=0.05) #TODO: not hardcoded
        self.lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        
        self.total_epochs = 0
        
        #Various intervals
        self.lr_interval = 10
        
        #Logging
        self.verbose = verbose #log to terminal too?
        self.batch_log_interval = 1
        self.checkpoint_interval = 50 #additional checkpoints in number of batches
    
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
    def crossval(self):
        self.model.eval()
        total_validation_loss = 0.0
        total_seen = 0
        for batch_idx, ((Qs,Ps,Ns),l) in enumerate(self.validation_set):
            Q_emb = self.model(Qs).detach()
            P_emb = self.model(Ps).detach()
            N_emb = self.model(Ns).detach()
            
            total_validation_loss += float(self.accuracy_function(Q_emb, P_emb, N_emb))
            total_seen += int(len(l))
        
        total_validation_loss /= float(total_seen)
        print("Crossval_error {}".format(total_validation_loss))
        wandb.log({"epoch_val_error":total_validation_loss},step=wandb.run.step)
        
        return total_validation_loss
    
        
    def train(self, n_epochs):
        
        
        for _ in range(n_epochs):
            self.total_epochs += 1
            
            epoch_average_batch_loss = 0.0;
            batchgroup_average_batch_loss = 0.0;
            
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
                batchgroup_average_batch_loss += float(batch_loss)
                #TODO: Add proper logging
                
                ## DEBUG: Monitor Norms
                overall_mean_norms = -1.0
                if True:
                    mqn = float(torch.norm(Q_embedding_vectors.detach(),dim=1).mean())
                    mpn = float(torch.norm(P_embedding_vectors.detach(),dim=1).mean())
                    mnn = float(torch.norm(N_embedding_vectors.detach(),dim=1).mean())
                    overall_mean_norms = float((mqn + mpn + mnn)/3.0)
                    print("mean Q,P,N norms {:.5f} {:.5f} {:.5f} ".format(mqn, mpn, mnn))
                
                
                #DEBUG
                if self.verbose and 0 == batch_idx % self.batch_log_interval:
                    print("batch ({}) loss {:.5f} time {:.3f} s/item".format(batch_idx,
                                                            float(batch_loss),
                                                            batch_time_per_item)
                        )
                wandb.log({"batch_loss":float(batch_loss),
                            "time_per_item":batch_time_per_item,
                            "embedding_mean_l2_norm":overall_mean_norms
                            })
                
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
            self.model.train(False)
            
            #Until now, this was actually total batch loss
            epoch_average_batch_loss /= batch_idx
            wandb.log({"epoch_average_batch_loss":epoch_average_batch_loss
                        })
            #TODO: log LR for this epoch.
            
            #TODO: any logging
            #TODO: any validation checking, any learning_schedule stuff.
            if 0 == self.total_epochs % self.lr_interval:
                self.model.eval()
                
                #TODO: blah. Too slow.
                #self.lr_schedule.step(epoch_average_batch_loss)
            
            #CROSSVALIDATION
            if None != self.validation_set:
                self.crossval()
                        
                        
            
            #Also save a checkpoint after every epoch
            self.create_checkpoint()
                

if __name__ == "__main__":
    import os
    import torch
    import torchvision
    
    #testing
    run_id = wandb.util.generate_id()
    run_id = "12164540.bw"
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
    model = models.create_model("dummy")
    if wandb.run.resumed:
        print("Resuming from checkpoint")
        model_pickle_file = wandb.restore("model_state.pt")
        model.load_state_dict( torch.load(model_pickle_file.name) )
    wandb.watch(model, log_freq=100) #Won't work if we restore the full object from pickle. :(
    
    
    print("load data")
    all_train = ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/")
    train_data, crossval_data, _ = ImageLoader.split_imagefolder(all_train, [0.3,0.1,0.6])
    print("create dataloader")
    tsdl = ImageLoader.TripletSamplingDataLoader(train_data,batch_size=200, num_workers=0)
    tsdl_crossval = ImageLoader.TripletSamplingDataLoader(crossval_data,batch_size=200, num_workers=0,shuffle=False)
    
    print("create trainer")
    test_trainer = Trainer(model, tsdl, tsdl_crossval)
    test_trainer.loss_fn = LossFunction.create_loss("normed")
    
    print("Begin training")
    test_trainer.train(100)
