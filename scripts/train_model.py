try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=sShapelyDeprecationWarning)
except:
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

import gc
import os
import time
import json

import detectron2.utils.comm as comm

# import some common libraries
import numpy as np
import torch

# import some common detectron2 utilities
from detectron2.config import LazyConfig, get_cfg
from detectron2.engine import launch

from deepdisc.data_format.register_data import register_data_set
from deepdisc.model.loaders import return_test_loader, return_train_loader
from deepdisc.model.models import return_lazy_model
from deepdisc.training.trainers import (
    return_evallosshook,
    return_lazy_trainer,
    return_optimizer,
    return_savehook,
    return_schedulerhook,
)
from deepdisc.utils.parse_arguments import dtype_from_args, make_training_arg_parser



def dp1_key_mapper(dataset_dict):
    '''
    args
        dataset_dict: [dict]
            A dictionary of metadata
    
    returns
        fn: str
            The filepath to the corresponding image
    
    '''
    fn = dataset_dict["filename"]

    return fn


def main(args):
    # Hack if you get SSL certificate error
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # Handle args
    output_dir = args.output_dir
    run_name = args.run_name  
    bs = args.batch_size

    # Get file locations   
    trainfile ="/home/g4merz/DP1/processed_data/train_dicts_new.json"
    #Don't have an evalulation dataset
    #evalfile =""

    cfgfile = args.cfgfile
    
    # Load the config
    cfg = LazyConfig.load(cfgfile)
    for key in cfg.get("MISC", dict()).keys():
        cfg[key] = cfg.MISC[key]

    # Register the data sets
    astrotrain_metadata = register_data_set(
        cfg.DATASETS.TRAIN, trainfile, thing_classes=cfg.metadata.classes
    )
    #astroval_metadata = register_data_set(
    #    cfg.DATASETS.TEST, evalfile, thing_classes=cfg.metadata.classes
    #)
    
    # Set the output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Set batch size
    cfg.SOLVER.IMS_PER_BATCH = bs
    cfg.dataloader.train.total_batch_size = bs

    # Tell the solver when to reduce the learning rate and how long to train
    cfg.SOLVER.STEPS=[e1,e2,e3]
    cfg.SOLVER.MAX_ITER = efinal

    # Iterations for 15, 25, 35, 50 epochs
    epoch = int(6778/bs)
    e1 = epoch * 15
    e2 = epoch * 25
    e3 = epoch * 35
    efinal = epoch * 50


    # Can change to this for debugging
    #epoch= 4
    #e1 = epoch * 5
    #e2 = epoch * 10
    #e3 = epoch * 15
    #efinal = epoch * 20

    # We aren't doing eval loss
    #val_per = epoch

    # Return the model - Keep Freeze=False
    # Freeze=False means we allow all weights to update.  
    model = return_lazy_model(cfg,freeze=False)

    #mapper reads in images from the json file
    mapper = cfg.dataloader.train.mapper(
            cfg.dataloader.imagereader, dp1_key_mapper, cfg.dataloader.augs
        ).map_data

    # loader batches the data to feed into the model
    loader = return_train_loader(cfg, mapper)
    
    # No eval set
    #eval_loader = return_test_loader(cfg, mapper)    
    
    # Keep this for detectron2 reasons
    cfg.optimizer.params.model = model
        
    # Set initial learning rate
    cfg.optimizer.lr = 0.001

    # Load the optimizer (it is defined in the config) 
    # Usually we use Multistep LR which changes the learning rate after fixed iterations
    optimizer = return_optimizer(cfg)

    # Save the model every 10 epochs
    saveHook = return_savehook(run_name,save_period=epoch*10)

    # Only use lossHook if we have evaluation data set
    #lossHook = return_evallosshook(val_per, model, eval_loader)
    
    # Scheduler needed to correctly reduce learning rate at pre-defined epochs
    schedulerHook = return_schedulerhook(optimizer)
    hookList = [schedulerHook, saveHook]
    #hookList = [lossHook, schedulerHook, saveHook]
    
    #trainer wraps all the previous objects we loaded
    trainer = return_lazy_trainer(model, loader, optimizer, cfg, hookList)
    #output loss values every set period
    trainer.set_period(epoch//2)
    #train for the given number of iterations
    trainer.train(0, efinal)
    #trainer.train(0,20)

    # Save the loss curves
    if comm.is_main_process():
        with open(os.path.join(output_dir,run_name) + "_losses.json", 'w') as json_file:
            json.dump(trainer.lossdict_epochs, json_file)
       

if __name__ == "__main__":
    args = make_training_arg_parser().parse_args()
    print("Command Line Args:", args)

    print("Training Model")
    t0 = time.time()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(
            args
        ),
    )

    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Took {time.time()-t0} seconds")
    
