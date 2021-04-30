import torch
import torch.optim

import numpy

import torch.distributed as dist

import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.blah = torch.nn.Sequential(
                        torch.nn.Linear(2,20),
                        torch.nn.Sigmoid(),
                        torch.nn.Linear(20,20),
                        torch.nn.Sigmoid(),
                        torch.nn.Linear(20,4),
                        torch.nn.Sigmoid()
                    )
    
    def forward(self, data):
        
        #DEBUG DDP does not actually split the data for us. :(
        #print("({}) : {} ".format(dist.get_rank(), data))
        #exit(1)
        return self.blah(data)


class DistTrainer(object):
    def __init__(self, model,
            dataloader,
            validation_set=None,
            g=1.0,rank=-1):
        self.model = model
        self.dataloader = dataloader
        self.validation_set = validation_set
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        
        #TODO
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.7) #TODO: not hardcoded
        #self.learning_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        
        self.total_epochs = 0
        
        #Logging
        self.batch_log_interval = 10
        
        self.rank = rank
        
    def train(self, n_epochs):
        
        
        for _ in range(n_epochs):
            self.total_epochs += 1
            
            for batch_idx, (x,y) in enumerate(self.dataloader):
                
                self.model.train(True)
                self.optimizer.zero_grad()
                
                y_hat = self.model(x)
                
                batch_loss = self.loss_fn(y_hat,y)
                batch_loss.backward()
                
                self.optimizer.step()
                
                
                batch_loss = float(batch_loss)
                if(batch_idx % 20 == 0):
                    print("({}) batch_loss {} ".format(self.rank, float(batch_loss)))
                
                #TODO: Any per-batch logging
                #END of loop over batches
            self.model.train(False)
            
            #TODO: any logging
            #TODO: any validation checking, any learning_schedule stuff.
            if self.total_epochs % 1 == 0 and self.rank == 0:
                model.eval()
                with torch.no_grad():
                    all_x, all_y = self.dataloader.dataset[:]
                    y_hat_all = model(all_x)
                    epoch_loss = float(self.loss_fn(y_hat_all, all_y))
                    print("({}) epoch_loss {}".format(self.rank, epoch_loss))

if __name__ == "__main__":
    import torch
    
    import torch.distributed as dist
    
    dist.init_process_group("mpi")
    mpi_rank = dist.get_rank()
    mpi_world_size = dist.get_world_size()
    print(" Init MPI, my rank {} of {}".format(mpi_rank, mpi_world_size))
    
    print("load data")
    x = torch.Tensor(numpy.load("data.npy"))
    y = torch.LongTensor(numpy.load("labels.npy"))
    
    ds = torch.utils.data.TensorDataset(x,y)
    dl = torch.utils.data.DataLoader(ds,batch_size=4,shuffle=False)
    
    print("create model")
    model = MyModel()
    model = torch.nn.parallel.DistributedDataParallel(model)
    
    print("create trainer")
    test_trainer = DistTrainer(model, dl, None,rank=mpi_rank)
    
    print("Begin training")
    test_trainer.train(10000)
