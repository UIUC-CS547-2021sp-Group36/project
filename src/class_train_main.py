import argparse

import wandb

import torch

import models
import LossFunction
import ClassifierTrainer
import data.ImageLoader as ImageLoader

def main(args):
    
    wandb.init(id=args.run_id if args.run_id is not None else wandb.util.generate_id(),
                resume=args.resume,
                entity='uiuc-cs547-2021sp-group36',
                project='image_similarity',
                group="debugging",
                tags=args.wandb_tags)
    wandb.config.update(args,allow_val_change=True)
                
    if wandb.run.resumed:
        print("Resuming...")
        
    print("create model")
    model = models.create_model(args.model)
    if wandb.run.resumed:
        print("Resuming from checkpoint")
        model_pickle_file = wandb.restore("model_state.pt")
        model.load_state_dict( torch.load(model_pickle_file.name) )
    
    if args.initial_weights is not None:
        print("Overriding weights with loaded weights")
        model_pickle_filename = args.initial_weights
        model.load_state_dict( torch.load(model_pickle_filename) )
        
    wandb.watch(model, log_freq=100) #Won't work if we restore the full object from pickle. :(
    
    print("load data")
    all_train = ImageLoader.load_imagefolder(args.dataset)
    train_data, crossval_data, _ = ImageLoader.split_imagefolder(all_train, args.split)
    print("create dataloader")
    tsdl = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size, num_workers=args.num_workers)
    tsdl_crossval = torch.utils.data.DataLoader(crossval_data,batch_size=args.batch_size, num_workers=args.num_workers,shuffle=False)
    
    print("create trainer")
    test_trainer = ClassifierTrainer.ClassifierTrainer(model, tsdl, tsdl_crossval,
                                    lr=args.lr, weight_decay=args.weight_decay
                                    )
    test_trainer.checkpoint_interval = args.checkpoint
    test_trainer.verbose = args.verbose
    
    print("Begin training")
    if args.epochs is not None and args.epochs >= 0:
        test_trainer.train(args.epochs)
    else:
        #Train forever
        while True:
            should_continue = test_trainer.train(1000)
            if not should_continue:
                break


if __name__ == "__main__":
    from util.cli import *
    
    arg_parser = argparse.ArgumentParser(description='Train an image similarity vector embedding')
    arg_parser.add_argument("--verbose","-v",action="store_true",default=False)
    
    wandb_group = arg_parser.add_argument_group("wandb","Arguments related to Weights and Biases")
    wandb_group.add_argument("--run_id",type=str)
    wandb_group.add_argument("--resume",action="store_true")
    wandb_group.add_argument("--wandb_tags",type=str_list,default=[])
    
    dataset_group = arg_parser.add_argument_group("data")
    dataset_group.add_argument("--dataset","-d",metavar="TINY_IMAGENET_ROOT_DIRECTORY",type=str,default="/workspace/datasets/tiny-imagenet-200/")
    dataset_group.add_argument("--split",metavar="TRAIN_PROPORTION,CROSSVAL_PROPORTION",type=check_datasplit,default=[0.8,0.1,1.0-0.9])
    dataset_group.add_argument("--num_workers",type=nonneg_int,default=0)
    
    model_group = arg_parser.add_argument_group("model")
    model_group.add_argument("--model",type=str,default="LowDNewModel")
    model_group.add_argument("--initial_weights",type=str,default=None)
    
    training_group = arg_parser.add_argument_group("training")
    training_group.add_argument("--epochs",metavar="N_epochs",type=int)
    training_group.add_argument("--subepoch_size",metavar="N_samples",type=int,default=None,help="The trainer will consier it an epoch once it sees N_samples data, regardless of batch size.")
    training_group.add_argument("--batch_size",type=pos_int,default=200)
    training_group.add_argument("--checkpoint","-c",type=int, default=50,help="Interval for extra checkpoints if doing very long epochs. Being depricated.")
    
    training_alg_group = arg_parser.add_argument_group("training_alg")
    training_alg_group.add_argument("--optimizer",type=str,choices=["Adam"],default="Adam")
    training_alg_group.add_argument("--lr",type=nonneg_float,default=0.0001)
    training_alg_group.add_argument("--weight_decay",type=nonneg_float,default=0.0001)
    
    training_loss_group = arg_parser.add_argument_group("training_loss")
    #training_loss_group.add_argument("--loss",type=str,default="normed")
    #training_loss_group.add_argument("--margin","-g",type=nonneg_float,default=1.0)
    
    
    
    
    args = arg_parser.parse_args()
    args.wandb_tags.append("classifier")
    args.wandb_tags.append("pretrain")
    
    main(args)
